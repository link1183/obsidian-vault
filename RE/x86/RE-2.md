---
id: RE-2
aliases:
  - re-2
tags:
  - re
  - x86
---

[[RE-TOC]]

# Techniques of reverse engineering

2 basics techniques of malware
Static analysis, dynamic analysis

There are 2 basic techniques used in reverse engineering :

## 1. Static analysis

Static analysis uses software to analyse a binary **without** running it.

## 2. Dynamic analysis

Dynamic analysis uses software to analyse a binary by **running** it.

The most popular tool today is [IDA](https://hex-rays.com/ida-free), which is a cross platform, multi-processor disassembler and debugger.

Other examples of such tools are :

- Hopper disassembler
- OllyDbg

A **disassembler** is a tool used to convert a binary written in Assembly, C, C++ or others into Assembly language instructions that we can then debug and manipulate.

## Notes

Reverse engineering is much more than just malware analysis. Another common example of reverse engineering is testing threats in systems, trying to find a breach.

[[RE-3]]
