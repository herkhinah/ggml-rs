// Build script and bindings generation modified from https://github.com/rustformers/llama-rs

use std::{collections::HashSet, env, path::PathBuf};

use bindgen::callbacks::ParseCallbacks;

const GGML_SOURCE_DIR: &str = "ggml/src";

#[derive(Debug)]
struct RenameEnumVariants;

impl ParseCallbacks for RenameEnumVariants {
    fn will_parse_macro(&self, _name: &str) -> bindgen::callbacks::MacroParsingBehavior {
        bindgen::callbacks::MacroParsingBehavior::Default
    }

    fn generated_name_override(
        &self,
        _item_info: bindgen::callbacks::ItemInfo<'_>,
    ) -> Option<String> {
        None
    }

    fn generated_link_name_override(
        &self,
        _item_info: bindgen::callbacks::ItemInfo<'_>,
    ) -> Option<String> {
        None
    }

    fn int_macro(&self, _name: &str, _value: i64) -> Option<bindgen::callbacks::IntKind> {
        None
    }

    fn str_macro(&self, _name: &str, _value: &[u8]) {}

    fn func_macro(&self, _name: &str, _value: &[&[u8]]) {}

    fn enum_variant_behavior(
        &self,
        _enum_name: Option<&str>,
        _original_variant_name: &str,
        _variant_value: bindgen::callbacks::EnumVariantValue,
    ) -> Option<bindgen::callbacks::EnumVariantCustomBehavior> {
        None
    }

    fn enum_variant_name(
        &self,
        enum_name: Option<&str>,
        original_variant_name: &str,
        _variant_value: bindgen::callbacks::EnumVariantValue,
    ) -> Option<String> {
        if let Some("enum gguf_type") = enum_name {
            return original_variant_name
                .strip_prefix("GGUF_TYPE_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_type") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_TYPE_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_prec") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_PREC_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_backend_type") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_BACKEND_TYPE_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_ftype") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_FTYPE_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_op") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_OP_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_unary_op") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_UNARY_OP_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_object_type") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_OBJECT_TYPE_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_log_level") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_LOG_LEVEL_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_cgraph_eval_order") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_CGRAPH_EVAL_ORDER_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_task_type") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_TASK_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_op_pool") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_OP_POOL_")
                .map(|s| s.to_string());
        }
        if let Some("enum ggml_sort_order") = enum_name {
            return original_variant_name
                .strip_prefix("GGML_SORT_")
                .map(|s| s.to_string());
        }
        None
    }

    fn item_name(&self, _original_item_name: &str) -> Option<String> {
        None
    }

    fn header_file(&self, _filename: &str) {}

    fn include_file(&self, _filename: &str) {}

    fn read_env_var(&self, _key: &str) {}

    fn blocklisted_type_implements_trait(
        &self,
        _name: &str,
        _derive_trait: bindgen::callbacks::DeriveTrait,
    ) -> Option<bindgen::callbacks::ImplementsTrait> {
        None
    }

    fn add_derives(&self, _info: &bindgen::callbacks::DeriveInfo<'_>) -> Vec<String> {
        vec![]
    }

    fn process_comment(&self, _comment: &str) -> Option<String> {
        None
    }

    fn field_visibility(
        &self,
        _info: bindgen::callbacks::FieldInfo<'_>,
    ) -> Option<bindgen::FieldVisibilityKind> {
        None
    }
}

fn generate_bindings() {
    let librs_path = PathBuf::from("src").join("lib.rs");

    let mut bbuilder = bindgen::Builder::default()
        .derive_copy(true)
        .derive_debug(true)
        .derive_partialeq(true)
        .derive_partialord(true)
        .derive_eq(true)
        .derive_ord(true)
        .derive_hash(true)
        .impl_debug(true)
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
        .merge_extern_blocks(true)
        .enable_function_attribute_detection()
        .enable_cxx_namespaces()
        .sort_semantically(true)
        .parse_callbacks(Box::new(RenameEnumVariants))
        .clang_args(["-std=gnu99",  "-x", "c", "-std=gnu11", "-Iggml/include/ggml/", "-Wno-error=implicit-function-declaration", "-Wno-error=int-conversion"])
        .header("ggml/include/ggml/ggml.h")
        .header("ggml/include/ggml/ggml-alloc.h")
        .header("ggml/include/ggml/ggml-backend.h")
        .header("ggml/src/ggml.c")
        .header("ggml/src/ggml-backend-impl.h")
        .allowlist_file("ggml/include/ggml/ggml.h")
        .allowlist_file("ggml/include/ggml/ggml-alloc.h")
        .allowlist_file("ggml/include/ggml/ggml-backend.h")
        .allowlist_file("ggml/src/ggml.c")
        .allowlist_file("ggml/src/ggml-backend-impl.h")
        // Suppress some warnings
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(unused)]")
        .raw_line("pub const GGMLSYS_VERSION: Option<&str> = option_env!(\"CARGO_PKG_VERSION\");")
        // Do not generate code for ggml's includes (stdlib)
        //.allowlist_file(ggml_header_path.to_string_lossy())
        //.allowlist_file(ggml_source_path.to_string_lossy());
        ;
    if cfg!(feature = "use_cmake") {
        if cfg!(feature = "cublas") || cfg!(feature = "hipblas") {
            let hfn = PathBuf::from(GGML_SOURCE_DIR).join("ggml-cuda.h");
            let hfn = hfn.to_string_lossy();
            bbuilder = bbuilder.header(hfn.clone()).allowlist_file(hfn);
        }
        if cfg!(feature = "clblast") {
            let hfn = PathBuf::from(GGML_SOURCE_DIR).join("ggml-opencl.h");
            let hfn = hfn.to_string_lossy();
            bbuilder = bbuilder.header(hfn.clone()).allowlist_file(hfn);
        }
        if cfg!(feature = "metal") {
            let hfn = PathBuf::from(GGML_SOURCE_DIR).join("ggml-metal.h");
            let hfn = hfn.to_string_lossy();
            bbuilder = bbuilder.header(hfn.clone()).allowlist_file(hfn);
        }
    }

    let bindings = bbuilder.generate().expect("Unable to generate bindings");
    bindings
        .write_to_file(librs_path)
        .expect("Couldn't write bindings");
}

fn main() {
    // By default, this crate will attempt to compile ggml with the features of your host system if
    // the host and target are the same. If they are not, it will turn off auto-feature-detection,
    // and you will need to manually specify target features through target-features.
    println!("cargo:rerun-if-changed=ggml");

    // If running on docs.rs, the filesystem is readonly so we can't actually generate
    // anything. This package should have been fetched with the bindings already generated
    // so we just exit  here.
    if env::var("DOCS_RS").is_ok() {
        return;
    }
    if cfg!(not(feature = "use_cmake")) {
        return build_simple();
    }
    build_cmake();
}

fn build_cmake() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    generate_bindings();

    // This silliness is necessary to get the cc crate to discover and
    // spit out the necessary stuff to link with C++ (and CUDA if enabled).
    let mut build = cc::Build::new();
    build.cpp(true).file("dummy/dummy.c");

    if cfg!(feature = "cublas") {
        build.cuda(true);
    } else if cfg!(feature = "hipblas") {
        println!("cargo:rerun-if-changed=ROCM_PATH");
        build.cpp(true);
    }
    build.compile("dummy");

    let rocm_path = if cfg!(feature = "hipblas") {
        Some(PathBuf::from(
            env::var("ROCM_PATH").unwrap_or_else(|_| String::from("/opt/rocm")),
        ))
    } else {
        None
    };

    let mut cmbuild = cmake::Config::new("ggml");
    cmbuild.build_target("ggml");
    if cfg!(feature = "no_k_quants") {
        cmbuild.define("LLAMA_K_QUANTS", "OFF");
    }
    if cfg!(feature = "cublas") {
        cmbuild.define("GGML_CUBLAS", "ON");
        cmbuild.define("LLAMA_CUBLAS", "ON");
    } else if cfg!(feature = "hipblas") {
        let rocm_path = rocm_path.as_ref().expect("Impossible: rocm_path not set!");
        let rocm_llvm_path = rocm_path.join("llvm").join("bin");
        cmbuild.define("LLAMA_HIPBLAS", "ON");
        cmbuild.define("CMAKE_PREFIX_PATH", rocm_path);
        cmbuild.define("CMAKE_C_COMPILER", rocm_llvm_path.join("clang"));
        cmbuild.define("CMAKE_CXX_COMPILER", rocm_llvm_path.join("clang++"));
    } else if cfg!(feature = "clblast") {
        cmbuild.define("LLAMA_CLBLAST", "ON");
    } else if cfg!(feature = "openblas") {
        cmbuild.define("LLAMA_BLAS", "ON");
        cmbuild.define("LLAMA_BLAS_VENDOR", "OpenBLAS");
    }
    if target_os == "macos" {
        cmbuild.define(
            "LLAMA_ACCELERATE",
            if cfg!(feature = "no_accelerate") {
                "OFF"
            } else {
                "ON"
            },
        );
        cmbuild.define(
            "LLAMA_METAL",
            if cfg!(feature = "metal") { "ON" } else { "OFF" },
        );
    }
    let dst = cmbuild.build();
    if cfg!(feature = "cublas") {
        println!("cargo:rustc-link-lib=cublas");
    } else if cfg!(feature = "hipblas") {
        let rocm_path = rocm_path.as_ref().expect("Impossible: rocm_path not set!");
        println!(
            "cargo:rustc-link-search={}",
            rocm_path.join("lib").to_string_lossy()
        );
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=rocblas");
        let mut build = cc::Build::new();
        build.cpp(true).file("dummy/dummy.c").object(
            PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set!"))
                .join("build")
                .join("CMakeFiles")
                .join("ggml-rocm.dir")
                .join("ggml-cuda.cu.o"),
        );
        build.compile("dummy");
    } else if cfg!(feature = "clblast") {
        println!("cargo:rustc-link-lib=clblast");
        println!(
            "cargo:rustc-link-lib={}OpenCL",
            if target_os == "macos" {
                "framework="
            } else {
                ""
            }
        );
    } else if cfg!(feature = "openblas") {
        println!("cargo:rustc-link-lib=openblas");
    }
    if target_os == "macos" {
        if cfg!(not(feature = "no_accelerate")) {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        if cfg!(feature = "metal") {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        }
    }
    println!("cargo:rustc-link-search=native={}/build/src", dst.display());
    println!("cargo:rustc-link-lib=static=ggml");
}

fn build_simple() {
    if cfg!(feature = "cublas") || cfg!(feature = "clblast") || cfg!(feature = "hipblas") {
        panic!("Must build with feature use_cmake when enabling BLAS!");
    }
    generate_bindings();

    let mut builder = cc::Build::new();
    let build = builder
        .files([
            PathBuf::from(GGML_SOURCE_DIR).join("ggml.c"),
            PathBuf::from(GGML_SOURCE_DIR).join("ggml-alloc.c"),
            PathBuf::from(GGML_SOURCE_DIR).join("ggml-backend.c"),
            #[cfg(not(feature = "no_k_quants"))]
            PathBuf::from(GGML_SOURCE_DIR).join("ggml-quants.c"),
        ])
        .include(PathBuf::from(GGML_SOURCE_DIR))
        .include("include");
    #[cfg(not(feature = "no_k_quants"))]
    build.define("GGML_USE_K_QUANTS", None);

    // This is a very basic heuristic for applying compile flags.
    // Feel free to update this to fit your operating system.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let is_release = env::var("PROFILE").unwrap() == "release";
    let compiler = build.get_compiler();

    match target_arch.as_str() {
        "x86" | "x86_64" => {
            let features = x86::Features::get();

            if compiler.is_like_clang() || compiler.is_like_gnu() {
                build.flag("-pthread");

                features.iter().for_each(|feat| {
                    build.flag(&format!("-m{feat}"));
                });
            } else if compiler.is_like_msvc() {
                if features.contains("avx2") {
                    build.flag("/arch:AVX2");
                } else if features.contains("avx") {
                    build.flag("/arch:AVX");
                }
            }
        }
        "aarch64" => {
            if compiler.is_like_clang() || compiler.is_like_gnu() {
                if std::env::var("HOST") == std::env::var("TARGET") {
                    build.flag("-mcpu=native");
                } else if &target_os == "macos" {
                    build.flag("-mcpu=apple-m1");
                    build.flag("-mfpu=neon");
                }
                build.flag("-pthread");
            }
        }
        _ => (),
    }

    if &target_os == "macos" {
        build.define("GGML_USE_ACCELERATE", None);
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if is_release {
        build.define("NDEBUG", None);
    }
    build.warnings(false);
    build.compile(GGML_SOURCE_DIR);
}

fn get_supported_target_features() -> HashSet<String> {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap()
        .split(',')
        .filter(|s| x86::RELEVANT_FLAGS.contains(s))
        .map(ToString::to_string)
        .collect::<HashSet<_>>()
}

mod x86 {
    use super::HashSet;

    pub const RELEVANT_FLAGS: &[&str] = &["fma", "avx", "avx2", "f16c", "sse3"];
    pub struct Features(HashSet<String>);

    impl std::ops::Deref for Features {
        type Target = HashSet<String>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Features {
        pub fn get() -> Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::env::var("HOST") == std::env::var("TARGET") {
                return Self::get_host();
            }
            Self(super::get_supported_target_features())
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub fn get_host() -> Self {
            Self(
                [
                    std::is_x86_feature_detected!("fma"),
                    std::is_x86_feature_detected!("avx"),
                    std::is_x86_feature_detected!("avx2"),
                    std::is_x86_feature_detected!("f16c"),
                    std::is_x86_feature_detected!("sse3"),
                ]
                .into_iter()
                .enumerate()
                .filter(|(_, exists)| *exists)
                .map(|(idx, _)| RELEVANT_FLAGS[idx].to_string())
                .collect::<HashSet<_>>(),
            )
        }
    }
}
