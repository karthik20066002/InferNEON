ENTRY(_start)

MEMORY
{
    /* QEMU virt RAM starts at 0x40000000; QEMU loads its DTB at the base,
       so link the kernel 1 MiB up (matches legacy ORIGIN). Loader addr must match. */
    RAM (rwx) : ORIGIN = 0x40100000, LENGTH = 127M
}

SECTIONS
{
    .text.boot : {
        KEEP(*(.text.boot))
    } > RAM

    .text : {
        *(.text .text.*)
    } > RAM

    .rodata : {
        *(.rodata .rodata.*)
    } > RAM

    .data : {
        *(.data .data.*)
    } > RAM

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(16);
    . += 0x4000;
    stack_top = .;
}
