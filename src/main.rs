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
mod fetch;
mod gate;
mod head;
mod infer;
mod layer;
mod nnmath;
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
    match boot_fetch_and_run() {
        Ok(n) => {
            uart::print("generated: ");
            uart::print_hex_u32(n as u32);
            uart::print(" tokens\n");
        }
        Err(_) => {
            uart::print("infer: FAIL\n");
        }
    }
    loop {
        crate::sync::park();
    }
}

/// Directory + resident pinning, then the wired generation loop.
/// Prints each generated token id as hex (host decodes via tokenizer.json).
fn boot_fetch_and_run() -> Result<usize, ()> {
    use storage::BlockStorage;
    let mut blk = storage::virtio::VirtioBlk::new();
    blk.init().map_err(|_| ())?;
    // layer bases from the LBA-0 table (16 u64 entries at byte 16)
    let mut lba0 = [0u8; 512];
    blk.read_blocks(crate::config::LAYER_TABLE_LBA, &mut lba0).map_err(|_| ())?;
    let mut bases = [0u64; 16];
    for li in 0..16usize {
        let o = 16 + li * 8;
        bases[li] = u64::from_le_bytes([
            lba0[o], lba0[o + 1], lba0[o + 2], lba0[o + 3], lba0[o + 4], lba0[o + 5],
            lba0[o + 6], lba0[o + 7],
        ]);
    }
    // expert stride from the header (bytes 8..12)
    let stride = u32::from_le_bytes([lba0[8], lba0[9], lba0[10], lba0[11]]) as u64;
    if stride == 0 || stride % crate::config::STORAGE_BLOCK_SIZE as u64 != 0 {
        uart::print("fail: stride\n");
        return Err(());
    }
    // Directory sits right after the expert region; derived, never assumed.
    let dir_lba = crate::config::dir_lba_after_experts(bases[0], stride);
    uart::print("dir lba: ");
    uart::print_hex_u64(dir_lba);
    uart::print("\n");
    match fetch::read_dir_at(&mut blk, dir_lba) {
        Ok(n) => {
            uart::print("dir entries: ");
            uart::print_hex_u32(n as u32);
            uart::print("\n");
        }
        Err(()) => {
            uart::print("fail: dir magic/count\n");
            return Err(());
        }
    }
    if let Err(()) = fetch::pin_resident(&mut blk) {
        uart::print("fail: pin_resident (tag/len/dtype mismatch or read error)\n");
        return Err(());
    }
    uart::print("resident pinned\n");
    if !crate::exec::selftest() {
        uart::print("exec selftest: FAIL\n");
        return Err(());
    }
    uart::print("exec selftest: PASS\n");
    let mut out = [0u32; 8];
    let n = infer::run(&mut blk, &bases, stride, &mut out)?;
    Ok(n)
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
