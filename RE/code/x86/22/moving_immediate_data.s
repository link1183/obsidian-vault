# moving_immediate_data: mov immediate data between register & memory

.section .data


.section .bss
  .lcomm buffer 1

.section .text
  .globl _start

_start:
  nop # debugging purposes

mov_immediate_data_to_register:
  movl $100, %eax
  movl $0x50, buffer
