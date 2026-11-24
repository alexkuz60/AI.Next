from __future__ import annotations

import app_next


def test_app_created() -> None:
    assert app_next.app is not None
    assert app_next.app.layout is not None
    assert len(app_next.app.callback_map) == 7


def test_server_exists() -> None:
    assert app_next.server is not None
