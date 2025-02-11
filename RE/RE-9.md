---
id: RE-9
aliases:
  - re-9
tags:
  - re
  - x86
---

[[RE-TOC]]

# The basics of x86 architecture

A computer application is just a table of machine instructions unique to the way the CPU deals with them.

The basic architecture is made of a CPU, memory and I/O (input/output) devices, all connected by a system bus.

## CPU

The CPU consists of 4 parts, which are:

1. The CU (Control Unit) - Retrieves and decodes instructions from the CPU and storing and retrieving data from memory.
2. Execution Unit - Where the execution of instructions happens.
3. Registers - Internal CPU memory location used as a temporary data storage.
4. Flags - Indicate events when the execution occurs.

Upon completion of the execution of an instruction, the CPU fetches the address of the next instruction from a special register, the EIP (instruction pointer), and uses it to fetch the next instruction.

We can immediately see that if we managed to control the flow of the EIP, we would be able to control the execution of the program to do whatever we wanted. This is a popular technique used by malware.

The entire fetch and execution process is ruled by the clock, a quartz oscillator that emits square-wave pulses at precise intervals.

[[RE-10]]
