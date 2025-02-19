#moving_data_between_registers_and_memory: mov data between regs and memory

.section .data
  constant:
    .int 10

.section .text
  .globl _start

_start:
  nop

mov_immediate_data_between_registers_and_memory:
  movl $777, %eax
  movl %eax, constant
