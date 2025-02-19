;moving_immediate_data: mov immediate data between registers & memory

section .data

section .bss
  buffer resb 1

section .text
  global _start

_start:
  nop

mov_immediate_data_to_register:
  mov eax, 100 ;mov 100 into eax
  mov byte[buffer], 0x50 ;mov 0x50 into the buffer memory location

exit:
  mov eax, 1
  mov ebx, 0
  int 0x80
