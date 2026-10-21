"""Public-array input schema and label-role validation."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import sparse

ROLES = ("train", "calibrate", "select", "test")


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_graph(path):
    """Load x, y, edge_index, optional ids, time_step, and Boolean role masks.

    Edges retain their supplied direction and multiplicity. For an undirected
    graph, supply both orientations. Unknown responses have value -1.
    """
    with np.load(path, allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    x = np.asarray(data["x"])
    y = np.asarray(data["y"], dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("x must be finite [n,d] and y must have shape [n].")
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("y must contain only -1, 0, and 1.")
    edges = np.asarray(data.get("edge_index", np.empty((2, 0))), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E].")
    if edges.size and (edges.min() < 0 or edges.max() >= len(x)):
        raise ValueError("An edge endpoint is outside the node range.")
    adjacency = sparse.csr_matrix((np.ones(edges.shape[1]), edges), shape=(len(x), len(x)))
    ids = np.asarray(data.get("ids", np.arange(len(x))))
    if ids.shape != y.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("ids must contain one unique identifier per node.")
    data.update(x=x, y=y, adjacency=adjacency, ids=ids)
    for public, internal in (("acquire", "select"), ("evaluate", "test")):
        if public in data and internal in data:
            raise ValueError("Use one naming convention for acquisition/evaluation masks.")
        if public in data:
            data[internal] = data[public]
    return data


def uniform_roles(y, dataset, draw, train_count=500, calibration_count=200, additional_count=50):
    """Original label-blind uniform role assignment used by the v16 fraud runs."""
    value = int(hashlib.sha256(("strong-fraud-v16|" + dataset + "|" + str(draw)).encode()).hexdigest()[:16], 16)
    order = np.random.default_rng(value).permutation(np.flatnonzero(y >= 0))
    if len(order) <= train_count + calibration_count + additional_count:
        raise ValueError("The requested role counts leave no evaluation nodes.")
    boundaries = (0, train_count, train_count + calibration_count, train_count + calibration_count + additional_count, len(order))
    masks = {}
    for name, begin, end in zip(ROLES, boundaries[:-1], boundaries[1:]):
        masks[name] = np.zeros(len(y), dtype=bool)
        masks[name][order[begin:end]] = True
    return masks


def roles_from_data(data, dataset, draw):
    present = [key in data for key in ROLES]
    if any(present) and not all(present):
        raise ValueError("Provide all four role masks or omit all four.")
    masks = {key: np.asarray(data[key], dtype=bool) for key in ROLES} if all(present) else uniform_roles(data["y"], dataset, draw)
    validate_roles(masks, data["y"])
    return masks


def validate_roles(masks, y):
    occupied = np.zeros(len(y), dtype=np.int64)
    for name in ROLES:
        mask = masks[name]
        if mask.shape != y.shape or mask.dtype != bool:
            raise ValueError("Role masks must be Boolean vectors of length n.")
        if np.any(mask & (y < 0)):
            raise ValueError("Unknown responses cannot be assigned to supervised roles.")
        occupied += mask
    if np.any(occupied > 1):
        raise ValueError("Training, calibration, acquisition, and evaluation roles overlap.")
    if not masks["train"].any() or not masks["calibrate"].any() or not masks["test"].any():
        raise ValueError("Training, calibration, and evaluation roles must be nonempty.")


def save_record(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
