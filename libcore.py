from __future__ import annotations

import numpy as np
import pandas as pd


def matchDiffDeep(deep: int, deep_mode: int) -> int:
    """Return the diff_deep index based on deep_mode.

    Args:
        deep: Current depth level.
        deep_mode: 0=stazis, 1=deep-Y (shift -1), 2=smart.

    Returns:
        Adjusted depth index for the diff_waves lookup.
    """
    if deep_mode == 1 and deep > 0:
        return deep - 1
    return deep


def calcWavesMinMax(w1: np.ndarray, w2: np.ndarray, deep: int, deep_mode: int) -> tuple[float, float]:
    """Compute the shared y-axis range for wave and diff plots."""
    diff_deep = matchDiffDeep(deep, deep_mode)
    y_min = float(min(w1[:, deep].min(), w2[:, diff_deep].min()))
    y_max = float(max(w1[:, deep].max(), w2[:, diff_deep].max()))
    return y_min, y_max


def createShapesAndAnnotations(
    track_zones: np.ndarray,
    y_min: float,
    y_max: float,
) -> tuple[list[dict], list[dict]]:
    """Build Plotly shapes and annotations for track transition zones."""
    shapes: list[dict] = []
    annotations: list[dict] = []
    number = 1
    for x in track_zones:
        shapes.append(
            dict(type="rect", x0=x - 1, x1=x + 1, y0=y_min, y1=y_max,
                 fillcolor="rgba(255,165,0,0.1)", line=dict(width=0.1))
        )
        shapes.append(
            dict(type="line", x0=x, x1=x, y0=y_min, y1=y_max,
                 line=dict(color="orange", width=1, dash="dot"))
        )
        annotations.append(
            dict(x=x, y=y_max, text=f"#{number}:x{x}", showarrow=True,
                 arrowhead=1, ax=0, ay=-40, bordercolor="orange", borderwidth=1,
                 bgcolor="rgba(30,30,30,0.5)", font=dict(color="white", size=14))
        )
        number += 1
    return shapes, annotations


def chancesToPercentes(chances: list[float]) -> np.ndarray:
    """Convert raw chance scores to percentages that sum to exactly 100."""
    arr_chances = np.array(chances, dtype=float)
    total = float(np.sum(arr_chances))
    if total <= 0:
        return np.zeros_like(arr_chances)

    raw = 100.0 * arr_chances / total
    scaled = raw * 10.0
    floors = np.floor(scaled)
    remainder = int(round(1000 - floors.sum()))
    if remainder > 0:
        order = np.argsort(-(scaled - floors))
        floors[order[:remainder]] += 1
    return floors / 10.0


def matchRelationMatrix(c_data: np.ndarray, r_data: np.ndarray) -> np.ndarray:
    """Build a 2x3 relation matrix describing the local shape of two waves."""
    mr = np.zeros((2, 3), dtype=np.int16)
    mr[0, :] = c_data
    mr[1, :] = r_data
    return mr


def add_matrix_if_not_exists(
    mr_list: list[np.ndarray],
    new_mr: np.ndarray,
    index: dict[bytes, str] | None = None,
) -> str:
    """Add a relation matrix to the list if not already present, return its marker name."""
    key = new_mr.tobytes()
    if index is not None and key in index:
        return index[key]

    for idx, mat in enumerate(mr_list):
        if np.array_equiv(mat, new_mr):
            name = f"M{idx}"
            if index is not None:
                index[key] = name
            return name

    mr_list.append(new_mr)
    name = f"M{len(mr_list) - 1}"
    if index is not None:
        index[key] = name
    return name


def format_markers_map_to_table(markers_map: dict[str, list[dict]], max_track: int) -> pd.DataFrame:
    """Convert the markers map into a pandas DataFrame (tracks x markers)."""
    markers = sorted(markers_map.keys(), key=lambda x: int(x[1:]))
    df = pd.DataFrame(index=range(max_track + 1), columns=markers)
    df.index.name = "track"

    for marker, entries in markers_map.items():
        for entry in entries:
            df.at[entry["track"], marker] = entry["pos"]

    return df.fillna("")


def make_markers_map_with_indices(mr_map: list[list[str]], max_track: int) -> pd.DataFrame:
    """Build a marker->position map for all tracks and format as a table."""
    result: dict[str, list[dict]] = {}

    for mr_index, mr_list in enumerate(mr_map):
        positions_map: dict[str, list[int]] = {}
        for pos, marker in enumerate(mr_list):
            positions_map.setdefault(marker, []).append(pos)

        for marker, positions in positions_map.items():
            result.setdefault(marker, []).append({
                "track": mr_index,
                "count": len(positions),
                "pos": positions,
            })

    return format_markers_map_to_table(result, max_track)
