from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
from typing import Iterator

try:
    from importlib import resources as _resources
except ImportError:  # pragma: no cover - Python < 3.7 fallback
    _resources = None

# importlib.resources only gained `.files()` in Python 3.9. On 3.7/3.8 the
# stdlib module exists but lacks the API we use, so fall back to the backport.
if _resources is None or not hasattr(_resources, "files"):
    try:
        import importlib_resources as _resources  # type: ignore
    except ImportError:  # pragma: no cover - legacy fallback
        _resources = None

def _normalize_resource(package: str, resource: str) -> str:
    normalized = str(resource).replace("\\", "/").strip("/")
    package_root = package.split(".")[0]
    prefix = package_root + "/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized


def _resource_ref(package: str, resource: str):
    if _resources is None:
        raise ImportError(
            "Package resources require importlib.resources or importlib_resources."
        )

    ref = _resources.files(package)
    normalized = _normalize_resource(package, resource)
    if not normalized:
        return ref

    for part in normalized.split("/"):
        ref = ref.joinpath(part)
    return ref


def _load_pkg_resources():
    try:
        import pkg_resources
    except ImportError as exc:  # pragma: no cover - setuptools not installed
        raise ImportError(
            "Package resources require importlib.resources, importlib_resources, "
            "or pkg_resources."
        ) from exc
    return pkg_resources


def resource_stream(package: str, resource: str):
    normalized = _normalize_resource(package, resource)
    if _resources is not None:
        return _resource_ref(package, normalized).open("rb")
    pkg_resources = _load_pkg_resources()
    return pkg_resources.resource_stream(package, normalized)


def resource_listdir(package: str, resource: str) -> list[str]:
    normalized = _normalize_resource(package, resource)
    if _resources is not None:
        return sorted(entry.name for entry in _resource_ref(package, normalized).iterdir())
    pkg_resources = _load_pkg_resources()
    return sorted(pkg_resources.resource_listdir(package, normalized))


@contextmanager
def resource_path(package: str, resource: str) -> Iterator[Path]:
    normalized = _normalize_resource(package, resource)
    if _resources is not None:
        with _resources.as_file(_resource_ref(package, normalized)) as path:
            yield path
        return
    pkg_resources = _load_pkg_resources()
    yield Path(pkg_resources.resource_filename(package, normalized))


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
