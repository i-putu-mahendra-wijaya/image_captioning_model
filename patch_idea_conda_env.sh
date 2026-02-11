#!/bin/bash
set -euo pipefail

TF_PY="$(conda run -n tf-clean python -c 'import sys; print(sys.executable)')"

cp .idea/misc.xml .idea/misc.xml.bak

python - <<PY
from pathlib import Path
import re

p = Path(".idea/misc.xml")
s = p.read_text(encoding="utf-8")

tf_py = r"""$TF_PY"""

# Replace only the project-jdk-name attribute value
s2 = re.sub(r'(project-jdk-name=")[^"]*(")', r"\g<1>"+tf_py+r"\g<2>", s)

p.write_text(s2, encoding="utf-8")
print("Patched:", p)
print("Interpreter:", tf_py)
PY
