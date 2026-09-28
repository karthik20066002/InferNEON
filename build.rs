//! build.rs — makes the storage block/erase width configurable at build time.
//!
//! `STORAGE_BLOCK_SIZE` env (bytes) flows into `config::STORAGE_BLOCK_SIZE`
//! via `cargo:rustc-env`. Default 512 = QEMU virtio-blk default sector.
//! Override: `STORAGE_BLOCK_SIZE=4096 cargo build --release`
//! or `just build 1 4096`.

fn main() {
    println!("cargo::rerun-if-env-changed=STORAGE_BLOCK_SIZE");
    println!("cargo::rerun-if-env-changed=INFERNEON_MODEL");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=linker.ld");
    let raw = std::env::var("STORAGE_BLOCK_SIZE").unwrap_or_else(|_| "512".to_string());
    let parsed: usize = raw.trim().parse().expect("STORAGE_BLOCK_SIZE must be an integer");
    assert!(parsed >= 512, "STORAGE_BLOCK_SIZE must be >= 512");
    assert!(
        parsed.is_power_of_two(),
        "STORAGE_BLOCK_SIZE must be a power of two"
    );
    println!("cargo:rustc-env=STORAGE_BLOCK_SIZE={parsed}");
    let model = std::env::var("INFERNEON_MODEL").unwrap_or_else(|_| "olmoe".to_string());
    assert!(
        model == "olmoe" || model == "qwen",
        "INFERNEON_MODEL must be 'olmoe' or 'qwen'"
    );
    println!("cargo:rustc-env=INFERNEON_MODEL={model}");
}
