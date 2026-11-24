from __future__ import annotations

import base64
import csv

import numpy as np
import pandas as pd
from scipy.signal.windows import gaussian

import config
import libcore as lc

# ── CSV loading helpers ──────────────────────────────────────────────────────

def _detect_delimiter(sample: str) -> str:
    """Detect the column delimiter from the first lines of a CSV file."""
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t"


def _parse_numeric_csv(text: str) -> np.ndarray:
    """Parse text into a 2-D integer array, raising a clear error on bad input."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("CSV пуст")

    delimiter = _detect_delimiter("\n".join(lines[:5]))
    try:
        arr = np.loadtxt(lines, delimiter=delimiter)
    except ValueError as exc:
        raise ValueError(
            "Ожидается числовая матрица: только целые числа, "
            "разделитель tab, comma или semicolon"
        ) from exc

    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if len(lines) == 1 else arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Ожидается двумерная матрица")
    return arr.astype(int)


def _load_csv_as_array(source: str | list[str], delimiter: str | None = None) -> np.ndarray:
    """Load CSV data from a file path or list-of-lines into a numpy int array."""
    if isinstance(source, list):
        text = "\n".join(source)
    else:
        with open(source, encoding="utf-8-sig") as handle:
            text = handle.read()

    if delimiter is not None:
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            arr = np.loadtxt(lines, delimiter=delimiter)
        except ValueError as exc:
            raise ValueError("Ожидается числовая матрица") from exc
        if arr.ndim == 1:
            arr = arr.reshape(1, -1) if len(lines) == 1 else arr.reshape(-1, 1)
        return arr.astype(int)

    return _parse_numeric_csv(text)


def readFileContent(content: str, sorted: bool) -> tuple[np.ndarray, int, int]:
    """Decode a base64-encoded CSV upload and return data as numpy array."""
    if not content or "," not in content:
        raise ValueError("Пустой файл")

    header, content_string = content.split(",", 1)
    if not header.startswith("data:"):
        raise ValueError("Некорректный формат загрузки")

    try:
        decoded = base64.b64decode(content_string, validate=True)
    except Exception as exc:
        raise ValueError("Некорректный base64") from exc

    if len(decoded) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Файл больше {config.MAX_UPLOAD_MB:g} МБ")

    arr_data = _parse_numeric_csv(decoded.decode("utf-8-sig"))
    matrix_max, max_order = arr_data.shape

    if sorted:
        arr_data = np.sort(arr_data, axis=1)

    return arr_data, matrix_max, max_order


# ── Track helpers ────────────────────────────────────────────────────────────

def setTrackList(track_num: int, max_nums: int, hist_back: int) -> list[dict]:
    """Build the track selector dropdown data list."""
    return [{"label": f"Track {i + 1}", "value": str(i + 1)} for i in range(max_nums)]


def autoSelectNextTrack(csv_data: np.ndarray, hist_back: int, order: int) -> int:
    """Determine the next track number based on the last row before the matrix window."""
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
    """Load the default CSV file and prepare the initial matrix slice."""
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
    """Return a window function array for FFT processing."""
    if window_type == "hamming":
        return np.hamming(N)
    if window_type == "hann":
        return np.hanning(N)
    if window_type == "gaussian":
        return gaussian(N, std=0.4 * N)
    return np.ones(N)


def prepare_wave_fft_data(
    levels: np.ndarray,
    window_type: str = "rectangular",
) -> dict[int, dict[str, np.ndarray]]:
    """Compute FFT and wave data for all depth levels."""
    max_x, max_deep = levels.shape

    masks = np.sign(levels)
    levels_float = levels.astype("float")

    window = get_fft_window(max_x, window_type)
    fft_full = np.fft.fft(levels * window[:, None], axis=0)
    log_fft = np.log2(np.abs(fft_full) + 1e-12)

    fft_y = log_fft - log_fft.min(axis=0, keepdims=True)
    masked = fft_y * masks

    fmin = masked.min(axis=0, keepdims=True)
    fmax = masked.max(axis=0, keepdims=True)
    denom = np.where((fmax - fmin) == 0, 1.0, fmax - fmin)
    normalized = (masked - fmin) / denom

    wmin = levels_float.min(axis=0, keepdims=True)
    wmax = levels_float.max(axis=0, keepdims=True)
    scaled = normalized * (wmax - wmin) + wmin

    positive = np.where(scaled >= 0, scaled, 0.0)
    negative = np.where(scaled < 0, scaled, 0.0)

    return {
        c: {"fft_y": np.vstack((positive[:, c], negative[:, c]))}
        for c in range(max_deep)
    }


# ── Track & matrix calculations ──────────────────────────────────────────────

def calcAllTracks(order_data: np.ndarray, matrix_size: int, max_nums: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one-hot and cumulative-sum track matrices."""
    order_data = np.asarray(order_data)
    if order_data.ndim != 1:
        raise ValueError("order_data должен быть одномерным")
    if order_data.size != matrix_size:
        raise ValueError(
            f"Длина окна {order_data.size} не совпадает с размером матрицы {matrix_size}"
        )
    if max_nums < 1:
        raise ValueError("max_nums должен быть положительным")
    if order_data.size and (order_data.min() < 1 or order_data.max() > max_nums):
        raise ValueError(f"Значения окна должны быть в диапазоне [1, {max_nums}]")

    mark_tracks = np.zeros((matrix_size, max_nums), dtype=np.int32)
    np.put_along_axis(mark_tracks, (order_data - 1)[:, None], 1, axis=1)
    sum_tracks = np.cumsum(mark_tracks, axis=0, dtype=np.int32)
    return mark_tracks, sum_tracks


def calcLeapPoints(passport: np.ndarray, leap_points: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build a heatmap of transition points and collect zone boundaries for all tracks."""
    heatmap_leaps = np.zeros_like(passport, dtype=float)
    max_track = len(leap_points)
    leaps_zones: list[np.ndarray] = []

    for tr in range(max_track):
        points = leap_points[tr]
        points_x = points[0, :]
        points_y = points[1, :]
        leaps_zones.append(points_x[1:-1])

        size_x = len(points_x) - 1
        for px in range(1, size_x):
            x = int(points_x[px])
            deep = int(points_y[px])
            heatmap_leaps[x, deep] = tr + 1

    heatmap_leaps = np.where(heatmap_leaps == 0, np.nan, heatmap_leaps)
    return heatmap_leaps.T, leaps_zones


def extract_track_xy(separate_data: np.ndarray, cumulative_data: np.ndarray, track_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) coordinates for a single track's line on the matrix."""
    tn = track_num - 1
    idx = np.flatnonzero(separate_data[:, tn])
    n = idx.size

    x = np.concatenate(([0], idx, [cumulative_data.shape[0] - 1]))
    y = np.zeros(2, dtype=int)
    if n > 0:
        y_vals = cumulative_data[idx, tn]
        y = np.concatenate(([0], y_vals, [y_vals[-1]]))

    leap_points = np.vstack((x, y))

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
    """Extract coordinates for all tracks."""
    n_tracks = cumulative_data.shape[1]
    all_tracks: list[np.ndarray] = []
    leap_points: list[np.ndarray] = []
    for i in range(n_tracks):
        arr_t, arr_l = extract_track_xy(separate_data, cumulative_data, i + 1)
        all_tracks.append(arr_t)
        leap_points.append(arr_l)
    return all_tracks, leap_points


def calcPassport(matrix_data: np.ndarray, max_deep: int) -> np.ndarray:
    """Compute the event histogram (passport) for the matrix."""
    matrix_data = np.asarray(matrix_data, dtype=np.int64)
    rows, cols = matrix_data.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, max_deep), dtype=int)
    if matrix_data.max() >= max_deep:
        raise ValueError("Значения матрицы выходят за пределы max_deep")

    flat = (np.arange(rows, dtype=np.int64)[:, None] * max_deep + matrix_data).ravel()
    return np.bincount(flat, minlength=rows * max_deep).reshape(rows, max_deep)


# ── Deep mode & wave helpers ─────────────────────────────────────────────────

def calcWavesForDeepLevels(passport: np.ndarray, max_deep: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute level and difference waves from the passport histogram."""
    next_val = np.zeros_like(passport)
    next_val[:, :-1] = passport[:, 1:]
    levels = passport - next_val
    diff_waves = passport - levels
    return levels, diff_waves


# ── Prognosis ────────────────────────────────────────────────────────────────

def matchPrognosis(
    diff_waves: np.ndarray,
    matrix_levels: np.ndarray,
    leaps_zones: list[np.ndarray],
    max_nums: int,
) -> tuple[list[float], pd.DataFrame]:
    """Evaluate track chances based on wave relation patterns at transition zones."""
    chances: list[float] = []
    mr_list: list[np.ndarray] = []
    mr_index: dict[bytes, str] = {}
    mr_map: list[list[str]] = []

    tracks_num = len(leaps_zones)
    max_deep = diff_waves.shape[1]

    for tr in range(tracks_num):
        track_zones = leaps_zones[tr]
        n_zones = len(track_zones)
        deep = 1
        signs_weight = 0
        mr_track_map: list[str] = []

        for zone in track_zones:
            if deep >= max_deep:
                break
            x = int(zone)
            rw = diff_waves[:, deep]
            cw = matrix_levels[:, deep]
            xp = max(x - 1, 0)
            xn = min(x + 1, len(rw) - 1)

            rp, rc, rn = rw[xp], rw[x], rw[xn]
            cp, cc, cn = cw[xp], cw[x], cw[xn]

            if not (cp == cc == cn) and not (rp == rc == rn):
                mr = lc.matchRelationMatrix(
                    np.array([cp, cc, cn]),
                    np.array([rp, rc, rn]),
                )
                mr_name = lc.add_matrix_if_not_exists(mr_list, mr, mr_index)
                mr_track_map.append(mr_name)
                signs_weight += 1

            deep += 1

        track_chance = signs_weight / n_zones if n_zones > 0 else 0.0
        chances.append(track_chance)
        mr_map.append(mr_track_map)

    mr_table = lc.make_markers_map_with_indices(mr_map, max_nums - 1)
    return chances, mr_table


# ── Inverse matrix ───────────────────────────────────────────────────────────

def calcInverseMatrix(
    input_arr: np.ndarray,
    matrix_size: int,
    max_nums: int,
    max_deep: int,
) -> np.ndarray:
    """Compute the inverse (future->past) event matrix."""
    _, arr_matrix_f = calcAllTracks(input_arr, matrix_size, max_nums)
    passport_f = calcPassport(arr_matrix_f, max_deep)
    matrix_f = np.where(passport_f == 0, np.nan, passport_f.astype(float))
    return matrix_f.T
