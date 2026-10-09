//! Chat prompt rendering through the model's own Jinja template.
//!
//! The template is read from the GGUF (`tokenizer.chat_template`) or supplied explicitly, exactly
//! like llama-server does. It is compiled and probed once at startup, so an unusable template
//! stops the service instead of silently producing prompts in the wrong format.
//!
//! `llama_chat_apply_template` is deliberately not used: it has no Jinja engine and only knows a
//! fixed list of template families.

use minijinja::{context, Environment, Error, ErrorKind};
use minijinja_contrib::pycompat;
use serde::Serialize;
use tracing::info;

use crate::engine::LlamaModel;
use crate::errors::AppError;

const TEMPLATE_NAME: &str = "chat";

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
        }
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Builds the template for `model`, preferring an explicit `override_source`.
    pub fn for_model(model: &LlamaModel, override_source: Option<&str>) -> Result<Self, AppError> {
        let source = override_source
            .or_else(|| model.chat_template())
            .ok_or_else(|| {
                AppError::ChatTemplateError(
                    "the model has no embedded chat template; provide one with --chat-template-file"
                        .to_string(),
                )
            })?;

        Self::new(
            source,
            model.bos_text().unwrap_or_default(),
            model.eos_text().unwrap_or_default(),
        )
    }

    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self, AppError> {
        let mut env = Environment::new();
        // HF templates are written against Python's Jinja2 and lean on dict/str methods.
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned(TEMPLATE_NAME, source.to_string())
            .map_err(|err| template_error("failed to compile", &err))?;

        let template = Self {
            env,
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
        };
        template.self_check()?;
        Ok(template)
    }

    /// Renders a conversation, ending with the assistant generation prompt.
    pub fn render(&self, messages: &[ChatMessage]) -> Result<String, AppError> {
        self.env
            .get_template(TEMPLATE_NAME)
            .and_then(|template| {
                template.render(context! {
                    messages => messages,
                    add_generation_prompt => true,
                    bos_token => &self.bos_token,
                    eos_token => &self.eos_token,
                })
            })
            .map_err(|err| template_error("failed to render", &err))
    }

    /// The template must render a probe conversation and actually include its content.
    fn self_check(&self) -> Result<(), AppError> {
        let probe = [
            ChatMessage::new("system", "PROBE_SYSTEM"),
            ChatMessage::new("user", "PROBE_USER"),
        ];
        let rendered = self.render(&probe)?;
        if !rendered.contains("PROBE_SYSTEM") || !rendered.contains("PROBE_USER") {
            return Err(AppError::ChatTemplateError(format!(
                "template ignores message content; probe rendered as {rendered:?}"
            )));
        }

        info!(probe = ?rendered, "Chat template self-check passed");
        Ok(())
    }
}

fn raise_exception(message: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

fn template_error(action: &str, err: &Error) -> AppError {
    AppError::ChatTemplateError(format!("{action}: {err:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMMA4_TEMPLATE: &str = include_str!("../../tests/fixtures/gemma4_chat_template.jinja");

    #[test]
    fn gemma4_template_renders_a_single_bos() {
        let template = ChatTemplate::new(GEMMA4_TEMPLATE, "<bos>", "<eos>").expect("template");
        let rendered = template
            .render(&[
                ChatMessage::new("system", "SYS"),
                ChatMessage::new("user", "USER"),
            ])
            .expect("render");

        assert_eq!(
            rendered,
            "<bos><|turn>system\nSYS<turn|>\n<|turn>user\nUSER<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn python_style_dict_access_is_supported() {
        let source = "{% for m in messages %}{{ m.get('content') }}|{% endfor %}";
        let template = ChatTemplate::new(source, "", "").expect("template");
        let rendered = template
            .render(&[ChatMessage::new("user", "hello")])
            .expect("render");
        assert_eq!(rendered, "hello|");
    }

    #[test]
    fn bos_and_eos_come_from_the_caller() {
        let source =
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}";
        let template = ChatTemplate::new(source, "<s>", "</s>").expect("template");
        let rendered = template
            .render(&[ChatMessage::new("user", "x")])
            .expect("render");
        assert_eq!(rendered, "<s>x</s>");
    }

    #[test]
    fn broken_syntax_is_rejected_at_startup() {
        let err = ChatTemplate::new("{% for m in messages %}", "", "").err();
        assert!(matches!(err, Some(AppError::ChatTemplateError(_))));
    }

    #[test]
    fn template_that_ignores_messages_is_rejected() {
        let err = ChatTemplate::new("static prompt", "", "").err();
        assert!(matches!(err, Some(AppError::ChatTemplateError(_))));
    }

    #[test]
    fn raise_exception_surfaces_the_template_message() {
        let err = ChatTemplate::new("{{ raise_exception('roles must alternate') }}", "", "")
            .err()
            .expect("must fail");
        assert!(err.to_string().contains("roles must alternate"));
    }
}
