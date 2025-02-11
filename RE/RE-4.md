---
id: RE-4
aliases:
  - re-4
tags:
  - re
  - x86
---

[[RE-TOC]]

# Introduction to x86 Assembly

The x86 Assembly language is a family of backwards-compatible Assembly languages which provide compatibility back to the Intel 8008 series of processors (introduced in April 1972). Those languages uses mnemonics to represent CPU instructions.

The Assembly language works on many different operating systems. The focus will here be the Linux Assembly Language with the Intel syntax.

x86 Assembly has 2 different choices of syntax.

- The AT&T syntax was dominant in the Unix world (AT&T was developed at AT&T labs, who made Unix).
- The Intel syntax was originally used for the documentation of the x86 platform and was dominant in the MS-DOS and Windows environments.

This document focuses on the Intel syntax.

We will also focus on the 32-bit architecture, as most malware will be written on that architecture to infect as many systems as possible, as a 32-bit binary will run fine on a 64-bit architecture.

[[RE-5]]
