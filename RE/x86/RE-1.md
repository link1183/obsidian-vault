---
id: RE-1
aliases:
  - re-1
tags:
  - re
  - x86
---

[[RE-TOC]]

# Goals of reverse engineering

The discussion of reverse engineering is closely tied to the concept of malware analysis.

When detecting malware files, it is critical to develop signatures to detect malware infections through the network.

There are 3 types of signatures :

1. **Host based signatures** are utilized to find malicious code on a target machine. They are also referred as indicators, which are used to identify filed created or modified by the malware.
2. **Antivirus signatures** focus on what the malware does rather than the make-up of the malware.
3. **Network signatures** are used to find malicious code by examining network traffic. Tools such as **WireShark** are often used for that purpose.

[[RE-2]]
