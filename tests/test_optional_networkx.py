import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python_without_networkx(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    script = textwrap.dedent(
        f"""
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "networkx" or name.startswith("networkx."):
                raise ModuleNotFoundError("No module named 'networkx'")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        {code}
        """
    )

    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_prepare_proteins_imports_without_networkx():
    result = _run_python_without_networkx(
        """
        import prepare_proteins

        assert hasattr(prepare_proteins, "proteinModels")
        assert hasattr(prepare_proteins, "tricks")
        """
    )

    assert result.returncode == 0, result.stderr


def test_networkx_backed_helpers_raise_clear_error_when_used():
    result = _run_python_without_networkx(
        """
        import pandas as pd
        import prepare_proteins

        df = pd.DataFrame({"AnSource": ["A"], "AnTarget": ["B"]})

        try:
            prepare_proteins.tricks._getBondTopology(df)
        except ImportError as exc:
            assert "networkx" in str(exc)
        else:
            raise AssertionError("Expected ImportError when networkx is unavailable")
        """
    )

    assert result.returncode == 0, result.stderr
