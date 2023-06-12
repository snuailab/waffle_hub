import subprocess
import sys
from pathlib import Path
from typing import Union


def _refine_script(script_path):
    """Make script file to be compatible with both on Windows and Linux."""
    with open(script_path) as f:
        script = f.read()
    script = script.replace("\r\n", "\n").replace("\\", "/")
    with open(script_path, "w") as f:
        f.writelines(script)


def run_python_file(file_path: Union[str, Path]):
    """Run python file as a subprocess."""
    file_path = Path(file_path)

    _refine_script(file_path)

    return subprocess.run(
        [sys.executable, str(file_path)],
        check=True,
    )
