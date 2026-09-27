import os
import sys

# Ensure root and src are on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmarks.benchmark_suite import main

if __name__ == "__main__":
    main()
