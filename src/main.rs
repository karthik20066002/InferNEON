//! InferNEON — bare-metal AArch64 MoE inference.
//!
//! Pipeline: token queue → layer counter → gating top-k → dispatch
//! (unordered set + barrier) → expert cache resolve → join on matmul
//! outputs → layer branch (output head / sample / drain).
//! Exec kernels (NEON + scalar) run; weight fetch is still TODO.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
#![allow(dead_code, unused_variables)]

mod addr;
mod attn;
mod boot;
mod cache;
mod combine;
mod config;
mod dispatch;
mod exec;
mod gate;
mod head;
mod layer;
mod pipeline;
mod storage;
mod sync;
mod token;
mod uart;

#[cfg(not(test))]
use core::panic::PanicInfo;

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // TODO: print panic location over UART once formatting exists.
    loop {
        crate::sync::park();
    }
}

/// Rust entry point, called from `_start` in boot.rs after BSS zero + stack setup.
#[no_mangle]
pub extern "C" fn kernel_main() -> ! {
    uart::print("InferNEON stub boot\n");
    uart::print("model: ");
    uart::print(crate::config::MODEL_NAME);
    uart::print("\n");
    storage_probe();
    pipeline::run_stub();
    loop {
        crate::sync::park();
    }
}

/// Bring up virtio-blk and validate the LBA-0 layer table.
/// Prints the outcome over UART; a bad table is a boot-stopping report
/// (park), never a silent misalignment.
fn storage_probe() {
    use storage::{check_layer_table_header, BlockStorage};
    let mut blk = storage::virtio::VirtioBlk::new();
    match blk.init() {
        Ok(()) => uart::print("storage: virtio-blk OK\n"),
        Err(_) => {
            uart::print("storage: virtio-blk INIT FAIL\n");
            return;
        }
    }
    let mut lba0 = [0u8; 512];
    match blk.read_blocks(crate::config::LAYER_TABLE_LBA, &mut lba0) {
        Ok(()) => {}
        Err(_) => {
            uart::print("storage: LBA0 read FAIL\n");
            return;
        }
    }
    let magic = u32::from_le_bytes([lba0[0], lba0[1], lba0[2], lba0[3]]);
    let num_layers = u32::from_le_bytes([lba0[4], lba0[5], lba0[6], lba0[7]]);
    let stride = u32::from_le_bytes([lba0[8], lba0[9], lba0[10], lba0[11]]);
    match check_layer_table_header(magic, num_layers, stride) {
        Some((layers, expert_stride)) => {
            uart::print("layer table: magic ok\n");
            uart::print("layers: ");
            uart::print_hex_u32(layers);
            uart::print("\nstride: ");
            uart::print_hex_u32(expert_stride);
            uart::print("\n");
            // Data-path proof: read layer 0's base from the table, fetch its
            // first block (expert 0's gate head), report the first word.
            // Host cross-checks this against the source safetensors bytes.
            let mut b = [0u8; 8];
            b.copy_from_slice(&lba0[16..24]);
            let base0 = u64::from_le_bytes(b);
            let mut head = [0u8; 512];
            match blk.read_blocks(base0 / 512, &mut head) {
                Ok(()) => {
                    uart::print("expert00 head: ");
                    uart::print_hex_u32(u32::from_le_bytes([head[0], head[1], head[2], head[3]]));
                    uart::print("\n");
                }
                Err(_) => uart::print("expert00 read FAIL\n"),
            }
        }
        None => {
            uart::print("layer table: BAD MAGIC ");
            uart::print_hex_u32(magic);
            uart::print(" (blank/unpacked stub disk?)\n");
        }
    }
}
