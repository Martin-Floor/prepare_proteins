import os
from pathlib import Path
import subprocess
import sys
import textwrap

from prepare_proteins._resources import resource_listdir, resource_path, resource_stream


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_resource_helpers_can_access_packaged_files():
    assert "md.mdp" in resource_listdir(
        "prepare_proteins", "prepare_proteins/scripts/md/gromacs/mdp"
    )

    with resource_stream(
        "prepare_proteins", "prepare_proteins/scripts/md/analysis/calculateDistances.py"
    ) as stream:
        contents = stream.read().decode("utf-8")
    assert "compute_distances" in contents

    with resource_path("prepare_proteins.scripts", "export_maestro_models.py") as path:
        assert path.is_file()
        assert path.name == "export_maestro_models.py"


def test_prepare_proteins_import_does_not_load_pkg_resources():
    result = _run_python(
        """
        import sys
        import prepare_proteins

        assert "pkg_resources" not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr
