# Аудит AI.Next

Дата: 2026-11-24. Аудит выполнен в режиме read-only, дефекты подтверждены запуском кода.

## 1. Подтверждённые дефекты

### P0 — падения и неверные расчёты

| ID | Место | Симптом | Подтверждение | Исправление |
|----|-------|---------|---------------|-------------|
| P0-1 | `app_next.py:506` | Загрузка CSV падала с `NameError: deep_mode` | Воспроизведено | `deep_mode` берётся из `id-file-props-store` |
| P0-2 | `libgraphs.py:193` | Matrix-F падал с `ValueError: truth value of an array is ambiguous` | Воспроизведено | Проверка `data_f is not None and len(data_f)` |
| P0-3 | `libdatas.py:595` | Прогноз всегда 0 %: `signs_weight` не увеличивался | `chances_sum=0.00` | Инкремент веса в условии зоны |
| P0-4 | `app_next.py:479` | Смена порядка → `IndexError` в `put_along_axis` | Воспроизведено | `max_nums` по всему файлу + валидация |
| P0-5 | `app_next.py:581` | `matrix_size + hist_back > rows` → `IndexError` | Воспроизведено | Клэмп окна и проверка `>= 2` |
| P0-6 | `libdatas.py:606` | При `x == 0` брался `rw[-1]` | Воспроизведено | Клэмп `x-1`/`x+1` |
| P0-7 | `libdatas.py:726` | `deep_mode` не влиял на стартовый график | JSON фигур совпадал | Передаётся реальный `deep_mode` |
| P0-8 | `libdatas.py:717` | Зоны всегда от трека 0 | Статически | `leaps_zones[track_num]` |
| P0-9 | `app_next.py:563` | Выбор трека ничего не пересчитывал | Только `State` | `id-track-select` стал `Input` |
| P0-10 | `app_next.py:322` | Таблица маркеров не обновлялась | Нет `Output` | `Output("id-mr-table","data")` |
| P0-11 | `app_next.py:89` | Карточка показывала стартовый `max_size` | `arr_size` не использовался | Используется `arr_size` |
| P0-12 | `app_next.py:547` | Вложенный `ActionIcon` и мутация `State` | Статически | Возвращается только иконка, props копируются |
| P0-13 | `app_next.py:693` | 3D открывался на сервере через `fig.show()` | Статически | `dmc.Modal` + `dcc.Graph` |
| P0-14 | `app_next.py:621` | Аннотация `dict \| type(no_update)` | Диагностика IDE | `dict \| Any` / `object` |
| P0-15 | `app_next.py:23` | `APP_DEBUG` по умолчанию `true` | Статически | По умолчанию `false` |
| P0-16 | `libdatas.py:17` | Нечисловые CSV давали криптовый `ValueError` | Воспроизведено | Валидация и понятное сообщение |

### P1 — мёртвый код, UX, архитектура

- `id-matrix-store` не использовался, добавлял ~114 КБ в ответ.
- `id-markov-deep` и `excluderMenu` были без обработчиков.
- `id-switch-sort` и `id-switch-calc-matrix-f` были только `State`.
- Не было `dcc.Loading`, `running=` и обработки ошибок.
- Циклический импорт `libdatas ↔ libgraphs`.
- Отладочные `print` в `libdatas.py`.
- `format_markers_map_to_table(max_track=19)` — жёсткие 20 строк.
- `createCombinedSubplot` терял заголовки subplot.
- `matchRelationMatrix` имел неиспользуемые параметры.
- Не было тестов, линтера, `pyproject.toml`, CI.
- Репозиторий: pack 241 МиБ, 131 файл `archive/` в индексе, `train_data.csv` 3.5 МБ.

### P2 — производительность

- `calcAllDatas`: 0.05–0.12 с на 200–250 строк, CPU не узкое место.
- Основная цена — payload: фигура 114–164 КБ × 2, `file_props` 110 КБ, `levels_store` 26–38 КБ.
- `calcPassport` через `np.add.at`, линейный поиск матриц, цикл по столбцам в FFT.

## 2. Выполненный рефакторинг

### Этап 0. Каркас качества

- `pyproject.toml` (ruff, pytest), `requirements-dev.txt`, `Makefile`.
- `tests/` — 36 тестов на `libcore`, `libdatas`, `libgraphs`, `pipeline`, `callbacks`, `app`.
- `docs/AUDIT.md` — этот документ.

### Этап 1. Исправления P0

- `handle_upload` больше не падает на `deep_mode`, валидирует файл и окно.
- `drawMatrix` безопасно проверяет `data_f`.
- `matchPrognosis` накапливает `signs_weight`, клэмпит `x-1`/`x+1`, строки таблицы по `max_nums`.
- `calcAllTracks` проверяет длину окна и диапазон значений.
- `handle_recalc` клэмпит `hist_back`, `matrix_size`, `track`, `deep`.
- `compute_core` использует `leaps_zones[track_num]` и реальный `deep_mode`.
- `id-track-select`, `id-switch-sort`, `id-switch-calc-matrix-f`, `id-markov-deep` стали `Input`.
- `id-mr-table` обновляется через `Output`.
- `fileInfoCard` использует `arr_size`.
- `setDiffMode` возвращает иконку, не вкладывая `ActionIcon`.
- 3D открывается в `dmc.Modal` с `dcc.Graph`.
- `APP_DEBUG=false` по умолчанию, `host`/`port` из env.
- CSV валидируется: только целые числа, лимит `MAX_UPLOAD_MB`.

### Этап 2. Архитектура

- `app_next.py` — тонкая точка входа; `app.py` — фабрика `create_app()`.
- `layout.py` — компоненты, `callbacks.py` — обработчики и регистрация.
- `libcore.py` — общие волновые функции; `libdatas.py` — математика; `libgraphs.py` — Plotly; `pipeline.py` — расчёт.
- Циклический импорт устранён.
- `state.py` — серверные LRU-кэши `FileStore` и `CalcCache`.
- `file_data` и `levels_store` больше не передаются в браузер: в `dcc.Store` только `upload_id`/`calc_id`.
- Один callback пересчёта с `ctx.triggered_id`; дублирующие `Output` помечены `allow_duplicate=True`.
- `dcc.Loading` вокруг графиков, `dmc.Alert` для ошибок.
- CSS вынесен в `assets/style.css`.

### Этап 3. Производительность

- `calcPassport` → `np.bincount`.
- `add_matrix_if_not_exists` → индекс по `tobytes()`.
- `prepare_wave_fft_data` векторизован по столбцам.
- Удалён `id-matrix-store`; payload страницы уменьшен примерно на 250 КБ.
- `np.loadtxt` с определением разделителя (tab/comma/semicolon).

### Этап 4. Репозиторий и CI

- `archive/`, `old_py/`, `train_data.csv`, `test_data.csv` убраны из индекса git (файлы остаются на диске).
- `libmatrix.py` → `experiments/libmatrix.py`.
- `audit_qwen.md`, `resume.txt`, `задание` → `docs/`.
- GitHub Actions: ruff + pytest.
- README: запуск, структура, env, тесты.

## 3. Проверки

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "import app_next"
APP_DEBUG=false .venv/bin/python app_next.py
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/
```

Результаты на 2026-11-24: `ruff` — 0 ошибок, `pytest` — 36 passed, `GET /` — 200,
`/_dash-dependencies` — 7 callback'ов, E2E upload — 7 трейсов и 12 строк таблицы,
E2E recalc с Matrix-F — 8 трейсов и 20 строк, ошибок нет.

## 4. Остаточные риски

- Серверный кэш рассчитан на один процесс; для нескольких workers нужен общий кэш (Redis/Flask-Caching).
- История git не переписана: pack остаётся большим, пока не выполнено `git filter-repo` (по отдельному подтверждению).
- `experiments/libmatrix.py` не покрыт тестами и не используется приложением.
