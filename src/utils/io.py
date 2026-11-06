import json
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML config files.") from exc
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def resolve_path(path, root=None):
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(root) if root else project_root()) / path
