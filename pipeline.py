from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import libdatas as ld
import libgraphs as lg


@dataclass(slots=True)
class CoreResult:
    matrix_size: int
    max_nums: int
    track_num: int
    deep_mode: int
    calc_f: bool
    mark_tracks: np.ndarray
    sum_tracks: np.ndarray
    passport_h: np.ndarray
    tracks_points: list[np.ndarray]
    leap_points: list[np.ndarray]
    heatmap_leaps: np.ndarray
    leaps_zones: list[np.ndarray]
    matrix_levels: np.ndarray
    diff_waves: np.ndarray
    levels_fft: dict[int, dict[str, np.ndarray]]
    chances: list[float]
    mr_table: pd.DataFrame
    matrix_h: np.ndarray
    matrix_f: np.ndarray | list
    calc_time: float


def compute_core(
    order_data: np.ndarray,
    matrix_size: int,
    max_nums: int,
    track_num: int,
    deep_mode: int,
    calc_f: bool = False,
) -> CoreResult:
    """Compute all numeric data for one matrix window (no Plotly figures)."""
    start = time.perf_counter()

    mark_tracks, sum_tracks = ld.calcAllTracks(order_data, matrix_size, max_nums)
    max_deep = int(np.max(sum_tracks)) + 1

    passport_h = ld.calcPassport(sum_tracks, max_deep)
    tracks_points, leap_points = ld.extract_all_tracks(mark_tracks, sum_tracks)
    heatmap_leaps, leaps_zones = ld.calcLeapPoints(passport_h, leap_points)
    matrix_levels, diff_waves = ld.calcWavesForDeepLevels(passport_h, max_deep)
    levels_fft = ld.prepare_wave_fft_data(matrix_levels)
    chances, mr_table = ld.matchPrognosis(diff_waves, matrix_levels, leaps_zones, max_nums)

    matrix_h = np.where(passport_h == 0, np.nan, passport_h.astype(float)).T
    matrix_f = (
        ld.calcInverseMatrix(order_data[::-1], matrix_size, max_nums, max_deep)
        if calc_f
        else []
    )

    return CoreResult(
        matrix_size=matrix_size,
        max_nums=max_nums,
        track_num=track_num,
        deep_mode=deep_mode,
        calc_f=bool(calc_f),
        mark_tracks=mark_tracks,
        sum_tracks=sum_tracks,
        passport_h=passport_h,
        tracks_points=tracks_points,
        leap_points=leap_points,
        heatmap_leaps=heatmap_leaps,
        leaps_zones=leaps_zones,
        matrix_levels=matrix_levels,
        diff_waves=diff_waves,
        levels_fft=levels_fft,
        chances=chances,
        mr_table=mr_table,
        matrix_h=matrix_h,
        matrix_f=matrix_f,
        calc_time=time.perf_counter() - start,
    )


def build_matrix_figure(
    core: CoreResult,
    subplots_balance: list[float],
    deep: int,
) -> object:
    """Build the combined waves/matrix figure for a computed core."""
    if not core.levels_fft:
        raise ValueError("Нет данных для построения графика")
    deep = max(0, min(int(deep), len(core.levels_fft) - 1))
    track_num = max(0, min(core.track_num, len(core.tracks_points) - 1))

    return lg.createCombinedSubplot(
        core.matrix_h,
        core.calc_f,
        core.matrix_f,
        core.tracks_points[track_num],
        track_num,
        core.heatmap_leaps,
        core.levels_fft,
        core.diff_waves,
        core.matrix_levels,
        core.leaps_zones[track_num],
        subplots_balance,
        deep,
        core.deep_mode,
    )


def build_chances_figure(core: CoreResult) -> object:
    """Build the prognosis bar figure for a computed core."""
    return lg.drawChancesBar(core.chances, core.max_nums, core.track_num)


def build_figures(
    core: CoreResult,
    subplots_balance: list[float],
    deep: int,
    deep_mode: int | None = None,
) -> tuple[object, object]:
    """Build both figures for a computed core."""
    if deep_mode is not None and deep_mode != core.deep_mode:
        core = replace(core, deep_mode=deep_mode)
    return (
        build_matrix_figure(core, subplots_balance, deep),
        build_chances_figure(core),
    )


def calcAllDatas(
    order_data: np.ndarray,
    matrix_size: int,
    max_nums: int,
    track_num: int,
    deep_mode: int,
    subplots_balance: list[float] | None = None,
    calc_f: bool = False,
    deep: int = 0,
) -> tuple[object, object, dict, pd.DataFrame, float]:
    """Backward-compatible pipeline entry point used by tests and scripts."""
    core = compute_core(order_data, matrix_size, max_nums, track_num, deep_mode, calc_f)
    fig_matrix, fig_chances = build_figures(
        core,
        subplots_balance or [0.5, 0.5],
        deep,
        deep_mode,
    )
    levels_store = {
        "diff_waves": core.diff_waves,
        "matrix_levels": core.matrix_levels,
        "levels_fft": core.levels_fft,
        "leaps_zones": core.leaps_zones,
    }
    return fig_matrix, fig_chances, levels_store, core.mr_table, core.calc_time
