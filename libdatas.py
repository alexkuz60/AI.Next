from __future__ import annotations

import time
import base64
import pprint
from typing import Any

import numpy as np
from scipy.signal.windows import gaussian
import pandas as pd

import libgraphs as lg


# ── CSV loading helpers ──────────────────────────────────────────────────────

def _load_csv_as_array(source: str | list[str], delimiter: str = '\t') -> np.ndarray:
    """Load CSV data from a file path or list-of-lines into a numpy int array.

    Args:
        source: Either a file path (str) or a list of CSV lines.
        delimiter: Column separator (default tab).

    Returns:
        2-D numpy array of int values.
    """
    if isinstance(source, list):
        arr_data = np.loadtxt(source, delimiter=delimiter).astype(int)
    else:
        with open(source, 'r', encoding='utf-8-sig') as f:
            arr_data = np.loadtxt(f, delimiter=delimiter).astype(int)
    return arr_data


def readFileContent(content: str, sorted: bool) -> tuple[np.ndarray, int, int]:
    """Decode a base64-encoded CSV upload and return data as numpy array.

    Args:
        content: Dash Upload component content string (with data-URI prefix).
        sorted: Whether to sort each column ascending.

    Returns:
        Tuple of (data_array, row_count, column_count).
    """
    content_type, content_string = str(content).split(',', 1)
    decoded = base64.b64decode(content_string)
    csv_lines = decoded.decode('utf-8-sig').splitlines()
    arr_data = _load_csv_as_array(csv_lines)
    matrix_max = arr_data.shape[0]
    max_order = arr_data.shape[1]

    if sorted:
        arr_data = np.sort(arr_data, axis=1)

    return arr_data, matrix_max, max_order


# ── Track helpers ────────────────────────────────────────────────────────────

def setTrackList(track_num: int, max_nums: int, hist_back: int) -> list[dict]:
    """Build the track selector dropdown data list.

    Args:
        track_num: Currently selected track (0-based).
        max_nums: Maximum number of tracks.
        hist_back: History lookback value (reserved for future use).

    Returns:
        List of dicts suitable for a DMC Select component.
    """
    return [{"label": f"Track {i+1}", "value": str(i+1)} for i in range(max_nums)]


def reshapeMatrix(csv_data: np.ndarray, matrix_data: np.ndarray, new_size: int, hist_back: int) -> np.ndarray:
    """Resize the matrix view by copying a slice from csv_data into matrix_data.

    Args:
        csv_data: Full source data array.
        matrix_data: Destination array (must be large enough).
        new_size: Number of rows to copy.
        hist_back: Row offset from the start.

    Returns:
        The destination array (modified in place).
    """
    np.copyto(matrix_data, csv_data[hist_back: new_size + hist_back])
    return matrix_data


def autoSelectNextTrack(csv_data: np.ndarray, hist_back: int, order: int) -> int:
    """Determine the next track number based on the last row before the matrix window.

    Args:
        csv_data: Full data array.
        hist_back: History lookback rows.
        order: Column index for the selected order.

    Returns:
        0-based track number.
    """
    if hist_back > 0:
        return int(csv_data[hist_back - 1, order]) - 1
    return 0


def uploadFileByName(
    file_name: str,
    order: int,
    matrix_size: int,
    hist_back: int,
    sorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    """Load the default CSV file and prepare the initial matrix slice.

    Args:
        file_name: Path to the CSV file.
        order: Column index for the selected order.
        matrix_size: Number of rows for the matrix.
        hist_back: History lookback rows.
        sorted: Whether to sort columns ascending.

    Returns:
        Tuple of (full_data, matrix_slice, matrix_size, max_size, max_order, max_nums, track_num).
    """
    csv_data = _load_csv_as_array(file_name)

    max_size = csv_data.shape[0] - hist_back
    max_order = csv_data.shape[1]
    max_nums = int(csv_data.max())

    if matrix_size > max_size:
        matrix_size = max_size

    if sorted:
        csv_data = np.sort(csv_data, axis=1)

    matrix_data = np.array(csv_data[hist_back: matrix_size + hist_back, :], dtype=int)

    track_num = autoSelectNextTrack(csv_data, hist_back, order)

    return csv_data, matrix_data, matrix_size, max_size, max_order, max_nums, track_num


# ── FFT helpers ──────────────────────────────────────────────────────────────

def get_fft_window(N: int, window_type: str) -> np.ndarray:
    """Return a window function array for FFT processing.

    Args:
        N: Window length.
        window_type: One of 'hamming', 'hann', 'gaussian', 'rectangular'.

    Returns:
        1-D numpy array of window coefficients.
    """
    if window_type == 'hamming':
        return np.hamming(N)
    elif window_type == 'hann':
        return np.hanning(N)
    elif window_type == 'gaussian':
        return gaussian(N, std=0.4 * N)
    else:
        return np.ones(N)  # rectangular window


def prepare_wave_fft_data(
    levels: np.ndarray,
    window_type: str = 'rectangular',
) -> dict[int, dict[str, np.ndarray]]:
    """Compute FFT and wave data for all depth levels.

    Applies a window function, computes log-scaled FFT magnitudes,
    masks positive/negative half-waves, and rescales FFT to wave amplitude range.

    Args:
        levels: 2-D array (n_rows, n_deep) of wave values.
        window_type: FFT window type.

    Returns:
        Dict mapping depth index → {'fft_y': 2-D array [pos, neg]}.
    """
    max_x, max_deep = levels.shape

    # Masks for positive/negative half-waves
    masks = np.sign(levels)
    levels_float = levels.astype('float')

    # Windowed FFT
    window = get_fft_window(max_x, window_type)
    win_track = levels * window[:, None]
    fft_full = np.fft.fft(win_track, axis=0)
    abs_fft = np.abs(fft_full)
    bias = 1e-12
    log_fft = np.log2(abs_fft + bias)

    data: dict[int, dict[str, np.ndarray]] = {}

    for c in range(max_deep):
        wave_y = levels_float[:, c]

        fft_y = log_fft[:, c].copy()
        fft_y -= np.min(fft_y)
        fft_y_masked = fft_y * masks[:, c]
        fft_y_masked = fft_y_masked.astype('float')

        fmin, fmax = np.min(fft_y_masked), np.max(fft_y_masked)
        wmin, wmax = np.min(wave_y), np.max(wave_y)
        denom = (fmax - fmin) or 1.0
        fft_y_masked = (fft_y_masked - fmin) / denom
        fft_y_masked = fft_y_masked * (wmax - wmin) + wmin

        pos_mask = (fft_y_masked >= 0)
        neg_mask = (fft_y_masked < 0)
        fft_y_2d = np.vstack((fft_y_masked * pos_mask, fft_y_masked * neg_mask))

        data[c] = {'fft_y': fft_y_2d}

    return data


# ── Track & matrix calculations ──────────────────────────────────────────────

def calcAllTracks(order_data: np.ndarray, matrix_size: int, max_nums: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one-hot and cumulative-sum track matrices.

    Args:
        order_data: 1-D array of event numbers for the selected order.
        matrix_size: Number of rows.
        max_nums: Maximum event number (= number of tracks).

    Returns:
        Tuple of (one-hot matrix, cumulative-sum matrix), both int32.
    """
    mark_tracks = np.zeros((matrix_size, max_nums), dtype=np.int32)
    np.put_along_axis(mark_tracks, (order_data - 1)[:, None], 1, axis=1)
    sum_tracks = np.cumsum(mark_tracks, axis=0, dtype=np.int32)
    return mark_tracks, sum_tracks


def calcLeapPoints(passport: np.ndarray, leap_points: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build a heatmap of transition points and collect zone boundaries for all tracks.

    Args:
        passport: Histogram matrix (n_rows, max_deep).
        leap_points: List of 2-D (x, y) coordinate arrays per track.

    Returns:
        Tuple of (heatmap with NaN for zero cells, list of zone x-coordinates per track).
    """
    heatmap_leaps = np.zeros_like(passport, dtype=float)
    max_track = len(leap_points)
    leaps_zones: list[np.ndarray] = []

    for tr in range(max_track):
        points = leap_points[tr]
        points_x = points[0, :]
        points_y = points[1, :]
        tr_zones = points_x[1:-1]
        leaps_zones.append(tr_zones)

        size_x = len(points_x) - 1
        for px in range(1, size_x):
            x = int(points_x[px])
            deep = int(points_y[px])
            heatmap_leaps[x, deep] = tr + 1

    # Replace zeros with NaN and transpose for plotting
    heatmap_leaps = np.where(heatmap_leaps == 0, np.nan, heatmap_leaps)
    heatmap_leaps = heatmap_leaps.T  # equivalent to rot90(k=1) + [::-1]

    return heatmap_leaps, leaps_zones


def extract_track_xy(separate_data: np.ndarray, cumulative_data: np.ndarray, track_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) coordinates for a single track's line on the matrix.

    Args:
        separate_data: One-hot track matrix.
        cumulative_data: Cumulative-sum track matrix.
        track_num: 1-based track number.

    Returns:
        Tuple of (expanded coords for drawing, compact coords for leap points), each shape (2, n).
    """
    tn = track_num - 1
    idx = np.flatnonzero(separate_data[:, tn])
    n = idx.size

    x = np.concatenate(([0], idx, [cumulative_data.shape[0] - 1]))
    y = np.zeros(2, dtype=int)
    if n > 0:
        y_vals = cumulative_data[idx, tn]
        y = np.concatenate(([0], y_vals, [y_vals[-1]]))

    leap_points = np.vstack((x, y))

    # Insert intermediate points for line drawing
    new_x: list[int] = []
    new_y: list[int] = []

    for i in range(len(x) - 1):
        if i > 0:
            new_x.append(int(x[i]) - 1)
            new_y.append(int(y[i]) - 1)
            new_x.append(int(x[i]))
            new_y.append(int(y[i]))
        else:
            new_x.append(int(x[i]))
            new_y.append(int(y[i]))

    new_x.append(len(separate_data) - 1)
    new_y.append(int(y[-1]))

    return np.vstack((new_x, new_y)), leap_points


def extract_all_tracks(
    separate_data: np.ndarray,
    cumulative_data: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract coordinates for all tracks.

    Args:
        separate_data: One-hot track matrix.
        cumulative_data: Cumulative-sum track matrix.

    Returns:
        Tuple of (list of expanded coords, list of compact leap-point coords).
    """
    n_tracks = cumulative_data.shape[1]
    all_tracks: list[np.ndarray] = []
    leap_points: list[np.ndarray] = []
    for i in range(n_tracks):
        arr_t, arr_l = extract_track_xy(separate_data, cumulative_data, i + 1)
        all_tracks.append(arr_t)
        leap_points.append(arr_l)
    return all_tracks, leap_points


def calcPassport(matrix_data: np.ndarray, max_deep: int) -> np.ndarray:
    """Compute the event histogram (passport) for the matrix.

    Counts occurrences of each event number (column value) at each row position.

    Args:
        matrix_data: 2-D array of event numbers.
        max_deep: Number of distinct event values (histogram bins).

    Returns:
        2-D histogram array (n_rows, max_deep).
    """
    rows = np.repeat(np.arange(matrix_data.shape[0]), matrix_data.shape[1])
    cols = matrix_data.ravel()
    passport = np.zeros((matrix_data.shape[0], max_deep), dtype=int)
    np.add.at(passport, (rows, cols), 1)
    return passport


# ── Deep mode & wave helpers ─────────────────────────────────────────────────

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
    """Compute the shared y-axis range for wave and diff plots.

    Args:
        w1: Diff waves array.
        w2: Matrix levels array.
        deep: Current depth level.
        deep_mode: Analysis mode (see matchDiffDeep).

    Returns:
        Tuple of (y_min, y_max).
    """
    diff_deep = matchDiffDeep(deep, deep_mode)
    y_min = float(min(w1[:, deep].min(), w2[:, diff_deep].min()))
    y_max = float(max(w1[:, deep].max(), w2[:, diff_deep].max()))
    return y_min, y_max


def createShapesAndAnnotations(
    track_zones: np.ndarray,
    y_min: float,
    y_max: float,
) -> tuple[list[dict], list[dict]]:
    """Build Plotly shapes and annotations for track transition zones.

    Args:
        track_zones: Array of x-coordinates for zone markers.
        y_min: Bottom of the y-axis range.
        y_max: Top of the y-axis range.

    Returns:
        Tuple of (list of shape dicts, list of annotation dicts).
    """
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


def calcWavesForDeepLevels(passport: np.ndarray, max_deep: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute level and difference waves from the passport histogram.

    Args:
        passport: Event histogram (n_rows, max_deep).
        max_deep: Number of depth levels (used to validate shape).

    Returns:
        Tuple of (matrix_levels, diff_waves), both shape (n_rows, max_deep).
    """
    next_val = np.zeros_like(passport)
    next_val[:, :-1] = passport[:, 1:]
    levels = passport - next_val
    diff_waves = passport - levels
    return levels, diff_waves


# ── Relation matrix helpers ──────────────────────────────────────────────────

def matchRelationMatrix(
    tr: int, deep: int, x: int, x_freq: int,
    c_data: np.ndarray, r_data: np.ndarray,
) -> np.ndarray:
    """Build a 2×3 relation matrix describing the local shape of two waves.

    Args:
        tr: Track index.
        deep: Current depth.
        x: Position index.
        x_freq: Frequency (distance from previous point).
        c_data: Cyan wave values [prev, curr, next].
        r_data: Red wave values [prev, curr, next].

    Returns:
        2×3 int16 array with relation markers.
    """
    mr = np.zeros((2, 3), dtype=np.int16)
    mr[0, :] = c_data
    mr[1, :] = r_data

    diff_c = mr[0, :].copy()
    diff_r = mr[1, :].copy()

    diff_c[1] = mr[0, 1] - mr[0, 0]
    diff_c[2] = mr[0, 2] - mr[0, 0]
    diff_c[0] = np.clip(diff_c[0], -3, 3)

    diff_r[1] = mr[1, 1] - mr[1, 0]
    diff_r[2] = mr[1, 2] - mr[1, 0]
    diff_r[0] = np.clip(diff_r[0], -3, 3)

    return mr


def add_matrix_if_not_exists(mr_list: list[np.ndarray], new_mr: np.ndarray) -> str:
    """Add a relation matrix to the list if not already present, return its marker name.

    Args:
        mr_list: Accumulated unique matrices.
        new_mr: Candidate matrix to add.

    Returns:
        Marker string like 'M0', 'M1', etc.
    """
    for idx, mat in enumerate(mr_list):
        if np.array_equiv(mat, new_mr):
            return f"M{idx}"
    mr_list.append(new_mr)
    return f"M{len(mr_list) - 1}"


def format_markers_map_to_table(markers_map: dict[str, list[dict]], max_track: int = 19) -> pd.DataFrame:
    """Convert the markers map into a pandas DataFrame (tracks × markers).

    Args:
        markers_map: Dict mapping marker name → list of entry dicts.
        max_track: Maximum track index for the table rows.

    Returns:
        DataFrame with track indices as rows and markers as columns.
    """
    markers = sorted(markers_map.keys(), key=lambda x: int(x[1:]))
    df = pd.DataFrame(index=range(max_track + 1), columns=markers)
    df.index.name = 'track'

    for marker, entries in markers_map.items():
        for entry in entries:
            df.at[entry['track'], marker] = entry['pos']

    return df.fillna('')


def make_markers_map_with_indices(mr_map: list[list[str]]) -> pd.DataFrame:
    """Build a marker→position map for all tracks and format as a table.

    Args:
        mr_map: List of marker-name lists, one per track.

    Returns:
        DataFrame of marker positions indexed by track.
    """
    result: dict[str, list[dict]] = {}

    for mr_index, mr_list in enumerate(mr_map):
        positions_map: dict[str, list[int]] = {}
        for pos, marker in enumerate(mr_list):
            positions_map.setdefault(marker, []).append(pos)

        for marker, positions in positions_map.items():
            if marker not in result:
                result[marker] = []
            result[marker].append({
                'track': mr_index,
                'count': len(positions),
                'pos': positions,
            })

    return format_markers_map_to_table(result)


# ── Prognosis ────────────────────────────────────────────────────────────────

def chancesToPercentes(chances: list[float]) -> np.ndarray:
    """Convert raw chance scores to percentages.

    Args:
        chances: List of raw scores per track.

    Returns:
        Array of percentages summing to 100.
    """
    arr_chances = np.array(chances, dtype=float)
    total = float(np.sum(arr_chances))
    if total > 0:
        prognosis = np.around(100 * arr_chances / total, decimals=1)
    else:
        prognosis = np.zeros_like(arr_chances)
    return prognosis


def matchPrognosis(
    diff_waves: np.ndarray,
    matrix_levels: np.ndarray,
    leaps_zones: list[np.ndarray],
) -> tuple[list[float], pd.DataFrame]:
    """Evaluate track chances based on wave relation patterns at transition zones.

    Args:
        diff_waves: Difference wave array.
        matrix_levels: Level wave array.
        leaps_zones: List of zone x-coordinates per track.

    Returns:
        Tuple of (chances list, marker DataFrame).
    """
    chances: list[float] = []
    mr_list: list[np.ndarray] = []
    mr_map: list[list[str]] = []

    tracks_num = len(leaps_zones)

    for tr in range(tracks_num):
        track_zones = leaps_zones[tr]
        max_deep = len(track_zones)
        deep = 1
        signs_weight = 0
        mr_track_map: list[str] = []
        x_prev = 0

        for x in track_zones:
            rw = diff_waves[:, deep]
            cw = matrix_levels[:, deep]
            x_freq = int(x - x_prev)
            x_prev = int(x)

            if x < len(rw) - 1:
                rp, rc, rn = rw[x - 1], rw[x], rw[x + 1]
                cp, cc, cn = cw[x - 1], cw[x], cw[x + 1]
            else:
                rn, cn = rw[x], cw[x]
                rp, rc = rn, cn  # fallback to avoid NameError
                cp, cc = cn, cn

            if not (cp == cc == cn) and not (rp == rc == rn):
                mr = matchRelationMatrix(tr, deep, x, x_freq,
                                         np.array([cp, cc, cn]),
                                         np.array([rp, rc, rn]))
                mr_name = add_matrix_if_not_exists(mr_list, mr)
                mr_track_map.append(mr_name)

            deep += 1

        track_chance = signs_weight / max_deep if max_deep > 0 else 0.0
        chances.append(track_chance)
        mr_map.append(mr_track_map)

    print("mr_count:", len(mr_list))
    mr_table = make_markers_map_with_indices(mr_map)

    return chances, mr_table


# ── Inverse matrix ───────────────────────────────────────────────────────────

def calcInverseMatrix(
    input_arr: np.ndarray,
    matrix_size: int,
    max_nums: int,
    max_deep: int,
) -> np.ndarray:
    """Compute the inverse (future→past) event matrix.

    Args:
        input_arr: Reversed order data array.
        matrix_size: Number of rows.
        max_nums: Maximum event number.
        max_deep: Maximum depth.

    Returns:
        2-D array with NaN for zero cells, ready for plotting.
    """
    _, arr_matrix_f = calcAllTracks(input_arr, matrix_size, max_nums)
    passport_f = calcPassport(arr_matrix_f, max_deep)
    matrix_f = np.where(passport_f == 0, np.nan, passport_f.astype(float))
    matrix_f = matrix_f.T  # equivalent to rot90(k=-1) for square-ish display
    return matrix_f


# ── Main pipeline ────────────────────────────────────────────────────────────

def calcAllDatas(
    order_data: np.ndarray,
    matrix_size: int,
    max_nums: int,
    track_num: int,
    deep_mode: int,
    subplots_balance: list[float] | None = None,
    calc_f: bool = False,
) -> tuple[Any, Any, dict, pd.DataFrame, float]:
    """Full calculation pipeline: tracks, passport, waves, FFT, prognosis, plots.

    Args:
        order_data: 1-D array of event numbers for the selected order.
        matrix_size: Number of rows in the matrix.
        max_nums: Maximum event number (= number of tracks).
        track_num: Selected track index (0-based).
        deep_mode: Analysis mode (0=stazis, 1=deep-Y, 2=smart).
        subplots_balance: Relative heights [upper, lower] for combined subplot.
        calc_f: Whether to also compute the inverse matrix.

    Returns:
        Tuple of (matrix figure, chances bar figure, levels store dict,
                  marker table DataFrame, computation time in seconds).
    """
    if subplots_balance is None:
        subplots_balance = [0.5, 0.5]

    start = time.time()

    mark_tracks, sum_tracks = calcAllTracks(order_data, matrix_size, max_nums)
    max_deep = int(np.max(sum_tracks)) + 1

    passport_h = calcPassport(sum_tracks, max_deep)

    tracks_points, leap_points = extract_all_tracks(mark_tracks, sum_tracks)
    heatmap_leaps, leaps_zones = calcLeapPoints(passport_h, leap_points)

    track = tracks_points[track_num]

    matrix_levels, diff_waves = calcWavesForDeepLevels(passport_h, max_deep)

    levels_fft = prepare_wave_fft_data(matrix_levels)
    levels_store: dict[str, Any] = {
        'diff_waves': diff_waves,
        'matrix_levels': matrix_levels,
        'levels_fft': levels_fft,
        'leaps_zones': leaps_zones,
    }

    chances, mr_table = matchPrognosis(diff_waves, matrix_levels, leaps_zones)
    chances_bar = lg.drawChancesBar(chances, max_nums, track_num)

    matrix_h = np.where(passport_h == 0, np.nan, passport_h.astype(float))
    matrix_h = matrix_h.T  # equivalent to rot90(k=1) + [::-1]

    matrix_f = calcInverseMatrix(order_data[::-1], matrix_size, max_nums, max_deep) if calc_f else []

    track_zones = leaps_zones[0]
    matrix_graf = lg.createCombinedSubplot(
        matrix_h, calc_f, matrix_f,
        track, track_num,
        heatmap_leaps,
        levels_fft, diff_waves, matrix_levels,
        track_zones,
        subplots_balance,
        deep=0,
        deep_mode=0,
    )

    calc_time = time.time() - start
    print(calc_time)

    return matrix_graf, chances_bar, levels_store, mr_table, calc_time
