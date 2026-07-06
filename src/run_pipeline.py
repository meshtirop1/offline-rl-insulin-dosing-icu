"""
End-to-end downstream driver: prepare RL tensors -> train CQL -> OPE.
Run after build_episodes.py has produced episodes.parquet. Kept as one script
so the whole chain survives as a single background job.
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent
PY = sys.executable

STEPS = [
    ("prepare_rl_dataset", [PY, str(SRC / "prepare_rl_dataset.py")]),
    ("train_cql",          [PY, str(SRC / "train_cql.py")]),
    ("evaluate_ope",       [PY, str(SRC / "evaluate_ope.py")]),
]

for name, cmd in STEPS:
    print(f"\n########## {name} ##########", flush=True)
    subprocess.run(cmd, check=True)

print("\n########## pipeline complete ##########")
