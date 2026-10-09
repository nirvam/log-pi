use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_INCLUDE_DIR");
    println!("cargo:rerun-if-changed=csrc/llama_bridge.h");
    println!("cargo:rerun-if-changed=csrc/llama_bridge.c");

    let mut cc_build = cc::Build::new();
    cc_build.file("csrc/llama_bridge.c");
    cc_build.include("csrc");

    if let Ok(inc_dir) = env::var("LLAMA_INCLUDE_DIR") {
        cc_build.include(inc_dir);
    } else {
        cc_build.include("/usr/include");
        cc_build.include("/usr/local/include");
    }

    cc_build.compile("llama_bridge");

    // Link underlying llama libraries
    if pkg_config::Config::new().probe("llama").is_ok() {
        return;
    }

    if let Ok(lib_dir) = env::var("LLAMA_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
    }

    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rustc-link-lib=ggml");
    println!("cargo:rustc-link-lib=ggml-base");
    println!("cargo:rustc-link-lib=stdc++");
}
