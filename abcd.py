from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent
scripts = sorted(
    p for p in ROOT.glob("*.py")
    if p.name[:2].isdigit() and p.name != Path(__file__).name
)

if not scripts:
    print("No experiment scripts found.")
    sys.exit(0)

for script in scripts:
    print(f"\n=== Running {script.name} ===")
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

print("\nAll experiments completed.")
