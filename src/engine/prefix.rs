//! Shared-prefix detection across tokenized prompts.

/// Length of the longest token prefix common to *all* prompts.
///
/// The result is capped at `min_len - 1`: every prompt must keep at least its final token for
/// itself, because the logits we read come from decoding that last token.
pub fn shared_prefix_len<T: AsRef<[i32]>>(prompts: &[T]) -> usize {
    let Some(first) = prompts.first().map(AsRef::as_ref) else {
        return 0;
    };

    let common = prompts[1..].iter().fold(first.len(), |common, other| {
        let matching = first
            .iter()
            .zip(other.as_ref())
            .take_while(|(a, b)| a == b)
            .count();
        common.min(matching)
    });
    let shortest = prompts.iter().map(|p| p.as_ref().len()).min().unwrap_or(0);

    common.min(shortest.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_prompts_have_no_prefix() {
        assert_eq!(shared_prefix_len::<Vec<i32>>(&[]), 0);
    }

    #[test]
    fn single_prompt_keeps_its_last_token() {
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3, 4]]), 3);
    }

    #[test]
    fn stops_at_the_first_divergence() {
        let prompts = [vec![1, 2, 3, 9], vec![1, 2, 3, 7, 8], vec![1, 2, 5, 5]];
        assert_eq!(shared_prefix_len(&prompts), 2);
    }

    #[test]
    fn identical_prompts_still_keep_a_suffix_token() {
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3], vec![1, 2, 3]]), 2);
    }

    #[test]
    fn a_prompt_that_is_a_prefix_of_another_is_capped() {
        assert_eq!(shared_prefix_len(&[vec![1, 2], vec![1, 2, 3, 4]]), 1);
    }

    #[test]
    fn disjoint_prompts_share_nothing() {
        assert_eq!(shared_prefix_len(&[vec![1, 2], vec![3, 4]]), 0);
    }
}
