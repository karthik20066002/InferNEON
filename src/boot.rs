//! boot.rs — `_start` stub: EL setup, per-core stacks, BSS zero, enter Rust.
//!
//! Design for N cores, bring up 1 for v1: primary (MPIDR_EL1 Aff0 == 0)
//! proceeds, secondaries park in `wfe`. Stack tops are carved per core from
//! `stack_top` (see linker.ld) — core count is NOT hardcoded in pipeline logic.

use core::arch::global_asm;

// `_start` only exists on the aarch64 target; host test builds skip it.
#[cfg(target_arch = "aarch64")]
global_asm!(
    r#"
    .section .text.boot, "ax"
    .global _start
    .type _start, %function
_start:
    // TODO: drop from EL2 to EL1 if entered at EL2; mask DAIF during early init.
    // Enable FP/SIMD for EL1 (CPACR_EL1.FPEN=0b11) BEFORE any f32/NEON.
    // Without this the first float insn traps EC=0x1F (was hidden while
    // the scaffold was integer-only UART).
    mrs x2, cpacr_el1
    movz x3, #0x30, lsl #16
    orr x2, x2, x3
    msr cpacr_el1, x2
    isb
    // Park-loop exception vectors: any fault halts the core instead of
    // fault-storming through a zeroed VBAR.
    ldr x2, =vectors
    msr vbar_el1, x2
    mrs x1, mpidr_el1
    and x1, x1, #0xff          // Aff0 = core index (virt: 0..N-1)
    cbz x1, 1f
0:  wfe                       // secondary cores park here (N-core later)
    b   0b
1:  // primary core:
    ldr x0, =stack_top
    mov sp, x0
    // zero BSS: [__bss_start, __bss_end)
    ldr x0, =__bss_start
    ldr x1, =__bss_end
2:  cmp x0, x1
    b.hs 3f
    str xzr, [x0], #8
    b   2b
3:  bl  kernel_main
4:  wfe
    b   4b

    // 16 x 0x80-spaced entries, each parks the core. Zero padding between
    // entries is never executed (each entry loops on itself).
    .balign 0x800
vectors:
    .rept 16
    .balign 0x80
5:  wfe
    b   5b
    .endr
    "#
);

extern "C" {
    static __bss_start: usize;
    static __bss_end: usize;
    static stack_top: usize;
}
