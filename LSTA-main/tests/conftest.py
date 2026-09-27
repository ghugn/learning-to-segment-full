import os
import sys

# Ensure both root and src/ are in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")

for path in [root_dir, src_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)
