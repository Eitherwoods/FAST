# -*- coding: utf-8 -*-
import re

s = open("main.tex", encoding="utf-8").read()
out = []
for line in s.split("\n"):
    m = re.match(r"\\section\{(.{0,24})", line)
    if m:
        out.append("\\typeout{==PG== \\thepage\\space " + m.group(1) + "}")
    out.append(line)
open("main.tex", "w", encoding="utf-8").write("\n".join(out))
print("marked")
