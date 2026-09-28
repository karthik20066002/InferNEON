//! uart.rs — PL011 UART stub (QEMU virt debug lifeline).
//! Only byte/hex output is real; formatting beyond that is TODO.

use crate::addr::UART0_BASE;

const UART_DR: usize = UART0_BASE;
const UART_FR: usize = UART0_BASE + 0x18;
const FR_TXFF: u32 = 1 << 5;

#[inline]
fn mmio_read(addr: usize) -> u32 {
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}

#[inline]
fn mmio_write(addr: usize, val: u32) {
    unsafe { core::ptr::write_volatile(addr as *mut u32, val) };
}

/// Blocking single-byte transmit. Real (needed for stub boot message).
pub fn putc(c: u8) {
    while mmio_read(UART_FR) & FR_TXFF != 0 {}
    mmio_write(UART_DR, c as u32);
}

/// Byte-wise string transmit. Real.
pub fn print(s: &str) {
    for b in s.bytes() {
        if b == b'\n' {
            putc(b'\r');
        }
        putc(b);
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Print u32 as `0x` + 8 hex digits. Real.
pub fn print_hex_u32(v: u32) {
    putc(b'0');
    putc(b'x');
    for i in (0..8).rev() {
        putc(HEX[((v >> (i * 4)) & 0xf) as usize]);
    }
}

/// Print u64 as `0x` + 16 hex digits. Real.
pub fn print_hex_u64(v: u64) {
    putc(b'0');
    putc(b'x');
    for i in (0..16).rev() {
        putc(HEX[((v >> (i * 4)) & 0xf) as usize]);
    }
}
