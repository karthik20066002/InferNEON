//! addr.rs — QEMU `virt` platform addresses (aarch64).

/// PL011 UART0.
pub const UART0_BASE: usize = 0x0900_0000;
/// virtio-mmio transport 0 (block device is attached here by justfile/qemu cmdline).
pub const VIRTIO_MMIO_BASE: usize = 0x0a00_0000;
/// GIC distributor / redistributor (TODO: needed only for N-core interrupts later).
pub const GICD_BASE: usize = 0x0800_0000;
/// RAM base for QEMU virt (kernel itself links 1 MiB up; see linker.ld).
pub const RAM_BASE: usize = 0x4000_0000;
