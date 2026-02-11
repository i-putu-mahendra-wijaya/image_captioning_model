#!/bin/bash
set -euo pipefail

TF_PY="$(conda run -n tf-clean python -c 'import sys; print(sys.executable)')"

cp .idea/misc.xml .idea/misc.xml.bak
cp .idea/image_captioning_model.iml .idea/image_captioning_model.iml.bak

python - <<PY
from pathlib import Path
import re

tf_py = r"""$TF_PY"""

def patch_attr(file_path, pattern, replacement):
    p = Path(file_path)
    s = p.read_text(encoding="utf-8")
    s2 = re.sub(pattern, replacement, s)
    p.write_text(s2, encoding="utf-8")
    print(f"Patched {file_path}")

patch_attr(
    ".idea/misc.xml",
    r'(project-jdk-name=")[^"]*(")',
    r"\g<1>" + tf_py + r"\g<2>",
)

patch_attr(
    ".idea/image_captioning_model.iml",
    r'(<orderEntry type="jdk" jdkName=")[^"]*(" jdkType="Python SDK" ?/>)',
    r"\g<1>" + tf_py + r"\g<2>",
)

print("Interpreter set to:", tf_py)
PY
