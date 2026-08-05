/* Minimal Cortex-M layout for the QEMU mps2-an386 board: flash at 0, RAM at
   0x20000000. Nothing here is specific to the estimator; it exists so the
   benchmark can be linked without pulling in a vendor board support package. */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(Reset_Handler)

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text       : { *(.text*) *(.rodata*) } > FLASH
  .data       : { *(.data*) } > RAM AT > FLASH
  .bss        : { *(.bss*) *(COMMON) } > RAM
  . = ALIGN(8);
  /* newlib's sbrk stub wants a heap base even though nothing here allocates. */
  end = .;
  _end = .;
  _estack = ORIGIN(RAM) + LENGTH(RAM);
}
