---
id: RE-7
aliases:
  - re-7
tags:
  - re
  - x86
---

[[RE-TOC]]

# Transistors and memory

Electronic computers are simply made of transistor switches.

Transistors are microscopic crystals of silicon that use electrical properties of the silicon to act as a switch. Modern computers have what are referred to as [field-effect transistors](https://en.wikipedia.org/wiki/Field-effect_transistor).

Let's take an example of 3 pins. When an electrical voltage is applied to pin 1, current flows between pins 2 and 3. When the voltage is removed from pin 1, current stops between pins 2 and 3.

Let's take an example of 3 pins. When an electrical voltage is applied to pin 1, current flows between pins 2 and 3. When the voltage is removed from pin 1, current stops between pins 2 and 3.

Zooming out a bit, there are also [diodes](https://en.wikipedia.org/wiki/Diode) and [capacitators](https://en.wikipedia.org/wiki/Capacitor), which when taken together with transistor switches, form a memory cell.

A memory cell will keep a minimum current flow to which when you apply a voltage on the input pin and a similar voltage on the select pin, a voltage will appear and remain on the output pin. The output voltage remains until the voltage is removed from the input pin **in conjunction** with the select pin.

[[RE-8]]
