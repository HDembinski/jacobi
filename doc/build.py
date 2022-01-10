#!/usr/bin/env python3
import subprocess as subp
from pathlib import Path
import sys

cmd = "sphinx-build -W -b html -d _build/doctrees . _build/html"

sys.exit(subp.call(cmd.split(), cwd=Path(__file__).parent))
