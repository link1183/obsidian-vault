---
id: RE-8
aliases:
  - re-8
tags:
  - x86
  - re
---

[[RE-TOC]]

# Bytes, Words, Double words

Memory is measured in bytes. A byte is 8 bits. Two bytes are called a word, two words are called a double word (4 bytes, 32 bits), also called a nibble, and a quad word is eight bytes (64 bits).

A bytes is 8 bits, which is 2\*\*8, which is 256. This means a byte can represent values from 0 to 255.

Every byte of memory in a computer has its own address, represented as a hexadecimal number.

For example, we can have an ESP register at address 0xffffd040 with value 0xf7fac3dc, both hexadecimal.

0xffffd040 is 4 bytes, a double word. That means it can represent values from 0 to 2\*\*32.

A computer program is nothing more than machine instructions stored in memory. A 32-bit CPU fetches a double word from memory address. A double word is 4 bytes in a row, read from memory. As soon as the CPU finishes executing the instruction, it fetches the next one from the instruction pointer.

[[RE-9]]
