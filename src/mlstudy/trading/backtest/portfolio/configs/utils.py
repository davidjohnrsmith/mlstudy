"""Config map utilities for portfolio sweep configs.

Maps short names to YAML file paths via ``sweep_config_map.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .sweep_config import PortfolioSweepConfig, load_sweep_config

_DEFAULT_CONFIG_MAP_PATH = Path(__file__).parent / "sweep_config_map.yaml"


def _resolve_config(
    config: str | Path | PortfolioSweepConfig,
    config_map_path: str | Path | None = None,
) -> PortfolioSweepConfig:
    """Resolve a config from a name, path, or existing object."""
    if isinstance(config, PortfolioSweepConfig):
        return config

    path = Path(config)
    # Heuristic: if it looks like a file path, load directly
    if path.suffix in (".yaml", ".yml") or "/" in str(config) or "\\" in str(config):
        return load_sweep_config(path)

    # Otherwise treat as a config-map name
    return load_sweep_config_by_name(str(config), config_map_path=config_map_path)


def load_config_map(path: str | Path | None = None) -> dict[str, str]:
    """Load a config map: ``{name: yaml_path}``.

    Parameters
    ----------
    path : str, Path, or None
        Path to the config map YAML.  If *None*, uses
        ``sweep_config_map.yaml`` next to this module.

    Returns
    -------
    dict[str, str]
    """
    path = Path(path) if path is not None else _DEFAULT_CONFIG_MAP_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config map must be a YAML mapping, got {type(raw).__name__}")
    return {str(k): str(v) for k, v in raw.items()}


def load_sweep_config_by_name(
    name: str,
    *,
    config_map_path: str | Path | None = None,
) -> PortfolioSweepConfig:
    """Load a sweep config by its short name from the config map.

    Parameters
    ----------
    name : str
        Config name (key in the config map).
    config_map_path : str, Path, or None
        Path to the config map YAML.  If *None*, uses the default
        ``sweep_config_map.yaml`` next to this module.

    Returns
    -------
    PortfolioSweepConfig
    """
    config_map = load_config_map(config_map_path)
    if name not in config_map:
        available = ", ".join(sorted(config_map)) or "(none)"
        raise KeyError(f"Config {name!r} not found in config map. Available: {available}")

    yaml_path = config_map[name]
    map_dir = (
        Path(config_map_path).parent
        if config_map_path is not None
        else _DEFAULT_CONFIG_MAP_PATH.parent
    )
    resolved = Path(yaml_path)
    if not resolved.is_absolute():
        resolved = map_dir / "tuning_configs" / resolved

    return load_sweep_config(resolved)
