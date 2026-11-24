from __future__ import annotations

import numpy as np
import pytest

import config
import libdatas as ld


@pytest.fixture(scope="session")
def csv_4x20() -> np.ndarray:
    return ld._load_csv_as_array(str(config.DEFAULT_CSV))


@pytest.fixture()
def small_matrix() -> np.ndarray:
    return np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ],
        dtype=int,
    )
