#!/usr/bin/env python3

import sys
import os

# Path to the 'src' directory relative to this file (tests/notebooks/path.py)
# os.path.dirname(__file__) is tests/notebooks/
# we go up two levels to get to the root (.. / ..)
# then we add 'src'
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'src'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
