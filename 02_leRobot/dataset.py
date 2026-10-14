"""LeRobot dataset reader for action-diffusion training.

Supports dataset v2 (one parquet per episode) and v3 (concatenated parquet files
plus ``meta/episodes``). Video features are left on disk; float observation
columns such as ``observation.state`` are the diffusion condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from action_diffusion.windows import WindowDataset, split_episode_mask


def resolve_root(repo_or_path: str) -> Path:
    path = Path(repo_or_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_or_path,
        repo_type="dataset",
        allow_patterns=["meta/**", "data/**"],
    )
    return Path(root)


def _read_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")
    return json.loads(info_path.read_text())


def _column_to_array(table: pa.Table, name: str) -> np.ndarray:
    column = table.column(name).combine_chunks()
    if pa.types.is_floating(column.type) or pa.types.is_integer(column.type):
        values = column.to_numpy(zero_copy_only=False).astype(np.float32)
        return values.reshape(-1, 1)
    rows = column.to_pylist()
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _float_observation_keys(info: dict) -> list[str]:
    keys = []
    for name, spec in info.get("features", {}).items():
        if not name.startswith("observation."):
            continue
        dtype = str(spec.get("dtype", ""))
        if dtype in {"float32", "float64", "float"}:
            keys.append(name)
    if not keys:
        raise ValueError("dataset has no float observation.* features")
    return keys


def _load_tables(paths: list[Path]) -> pa.Table:
    tables = [pq.read_table(path) for path in paths]
    return pa.concat_tables(tables, promote_options="default")


def _load_v3(root: Path, info: dict) -> tuple[pa.Table, np.ndarray]:
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no v3 data parquet under {root / 'data'}")
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"no v3 episode parquet under {root / 'meta' / 'episodes'}")
    data = _load_tables(data_files)
    episodes = _load_tables(episode_files).to_pandas().sort_values("episode_index")
    if "index" in data.column_names:
        order = np.argsort(data.column("index").combine_chunks().to_numpy(zero_copy_only=False), kind="stable")
        data = data.take(pa.array(order))
    episode_ends = episodes["dataset_to_index"].to_numpy(dtype=np.int64)
    return data, episode_ends


def _load_v2(root: Path, info: dict) -> tuple[pa.Table, np.ndarray]:
    chunks_size = int(info.get("chunks_size", 1000))
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing {episodes_path}")
    episodes = []
    for line in episodes_path.read_text().splitlines():
        if line.strip():
            episodes.append(json.loads(line))
    episodes.sort(key=lambda row: int(row["episode_index"]))
    tables = []
    ends = []
    cursor = 0
    for row in episodes:
        episode_index = int(row["episode_index"])
        chunk = episode_index // chunks_size
        path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(path)
        tables.append(table)
        cursor += table.num_rows
        ends.append(cursor)
    return pa.concat_tables(tables, promote_options="default"), np.asarray(ends, dtype=np.int64)


def load_lerobot_arrays(
    root: Path,
    obs_keys: list[str] | None = None,
    action_key: str = "action",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    info = _read_info(root)
    version = str(info.get("codebase_version", "v2.0"))
    data_path = str(info.get("data_path", ""))
    if version.startswith("v3") or "file-" in data_path:
        table, episode_ends = _load_v3(root, info)
    else:
        table, episode_ends = _load_v2(root, info)

    if obs_keys is None:
        obs_keys = _float_observation_keys(info)
    missing = [key for key in obs_keys + [action_key] if key not in table.column_names]
    if missing:
        raise KeyError(f"columns {missing} not in dataset. Have {table.column_names}")

    obs = np.concatenate([_column_to_array(table, key) for key in obs_keys], axis=-1)
    action = _column_to_array(table, action_key)
    if episode_ends[-1] != len(obs):
        raise ValueError(
            f"episode index range ends at {episode_ends[-1]} but loaded {len(obs)} frames"
        )
    return obs.astype(np.float32), action.astype(np.float32), episode_ends, obs_keys


def make_lerobot_datasets(
    repo_or_path: str,
    *,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    val_ratio: float = 0.02,
    seed: int = 42,
    obs_keys: list[str] | None = None,
    action_key: str = "action",
) -> tuple[WindowDataset, WindowDataset, Path, list[str]]:
    root = resolve_root(repo_or_path)
    obs, action, episode_ends, used_keys = load_lerobot_arrays(root, obs_keys, action_key)
    train_mask = split_episode_mask(len(episode_ends), val_ratio, seed)
    common = dict(
        obs=obs,
        action=action,
        episode_ends=episode_ends,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    train_set = WindowDataset(episode_mask=train_mask, **common)
    val_set = WindowDataset(episode_mask=~train_mask, **common)
    return train_set, val_set, root, used_keys
