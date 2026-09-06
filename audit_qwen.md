# Аудит кода проекта AI.Next

**Дата:** 2026-09-06  
**Проект:** Dash/Plotly приложение для анализа и прогнозирования событийных данных

---

## 1. Структура проекта

```
AI.Next/
├── app_next.py          # Основное приложение (Dash) — ~500 строк
├── libdatas.py          # Логика данных: FFT, матрицы, расчёты — ~680 строк
├── libgraphs.py         # Визуализация: Plotly графики — ~330 строк
├── libmatrix.py         # Эксперимент: матрицы характеристик волн — ~130 строк
├── requirements.txt     # 34 зависимости
├── csv_data/            # Входные данные (3 CSV файла)
├── old_py/              # Пустая директория (архив удалён)
├── archive/             # Архивные файлы
└── test_data.csv, train_data.csv  # Тестовые/обучающие данные
```

---

## 2. Архитектурные замечания

### 2.1 Глобальные переменные на уровне модуля (app_next.py)

При старте приложения (строки 15–41) выполняются тяжёлые вычисления:
- `ld.uploadFileByName()` — загрузка и парсинг CSV
- `ld.calcAllDatas()` — полный расчёт матрицы, FFT, маркеров

**Проблема:** эти вычисления блокируют запуск сервера. При изменении любого параметра по умолчанию потребуется перезапуск приложения.

**Рекомендация:** вынести первичную загрузку в callback или `dcc.Store` с `prevent_initial_call=False`.

### 2.2 Монолитные callback-функции

Callback `recalcMatrix` (~30 строк) и `uploadFile` (~25 строк) вызывают `ld.calcAllDatas()`, которая выполняет ~15 шагов расчёта. Один callback = один полный перерасчёт всего.

**Рекомендация:** разделить на инкрементальные обновления (отдельно пересчёт матрицы, отдельно обновление графиков).

### 2.3 Жёсткая привязка к файлу

```python
file_name = 'csv_data/4x20.csv'  # строка 22, hardcoded
```

**Рекомендация:** вынести в конфигурацию или `.env`.

---

## 3. Проблемы качества кода

### 3.1 libdatas.py

#### Dead code / закомментированные блоки

- **Строки 89–109** (`calcIntentVector`) — функция вычисляет координаты вектора намерений, но вызывается только из закомментированного блока в `calcLeapPoints` (строки 200–215). Функция не используется в рабочем потоке.
- **Строки 200–215** — большой закомментированный блок в `calcLeapPoints`, содержащий логику `vector_x`, `vector_y`, `tr_counter`.
- **Строка 300** — `matchDiffDeep`: при `deep_mode == 0` всегда возвращает 0; ветка `deep_mode == 1 and deep > 0` недостижима (тот же `deep_mode == 1` уже покрыт первым `if`). Логическая ошибка.

#### Дублирование логики

- `uploadFileByName` и `readFileContent` делают практически одно и то же (загрузка CSV → numpy array). Различаются только источником (файл vs base64-строка). Можно выделить общую функцию `_parse_csv_to_array`.

#### Магические числа

- `bias = 1e-12` (строка 138) — без пояснения
- `deep_mode` принимает значения 0, 1, 2 — не задокументировано что означают
- `subplots_balance=[0.5, 0.5]` — жёстко задано как значение по умолчанию, но ожидается список из двух элементов

#### Путаница с типами в `prepare_wave_fft_data`

- `wave_ds` создаётся через `np.repeat` + слайс, что даёт идентичный `levels`. Затем `wave_ds_float = wave_ds.astype('float')` — избыточно.
- `masks = np.sign(wave_ds)` — но позже используется `masks[:, c]` где `c` — индекс глубины, хотя `masks` имеет shape `(max_x, max_deep)`.

### 3.2 app_next.py

#### Пустой callback

```python
@callback(
    Input('id-switch-calc-matrix-f', 'checked')
)
def recalcAllDatasForReverseMatrix(on_off):
    print(on_off)
```
**Строки 464–467.** Callback ничего не делает, кроме `print()`. Нет Output — это нарушает контракт Dash callback.

**Рекомендация:** либо реализовать функционал, либо удалить.

#### Неправильный Output в callback

```python
@callback(
    Output('id-foot-drawer', 'opened'),
    Input('id-but-view-foot-drawer', 'n_clicks'),
    prevent_initial_call=True,
)
def showFootDrawer(click):
    return True  # Всегда True, drawer никогда не закроется
```
Drawer открывается по клику, но нет механизма закрытия.

#### Неправильное использование `persistence=True`

`trackSelector` и `id-slider-balance` имеют `persistence=True`, но после перерасчёта (`recalcMatrix`) значение трека перезаписывается через `Output('id-track-select', 'value')`. Это приводит к конфликту между сохранённым и вычисленным значением.

#### Глобальные константы, изменяемые в runtime

Переменные `order`, `hist_back`, `matrix_size`, `deep_mode`, `deep`, `sorted` объявлены как глобальные константы (строки 10–16), но дублируются в `dcc.Store` (`id-file-props-store`). Два источника истины.

### 3.3 libgraphs.py

#### Функция `draw3D`

```python
def draw3D(matrix_h):
```
Принимает параметр `matrix_h`, но в `app_next.py` вызывается как `lg.draw3D(data)` где `data = np.array(levels['diff_waves'].copy())`. Название параметра misleading.

#### `drawMatrix` — неиспользуемая функция?

Функция `drawMatrix` вызывается только из `createCombinedSubplot`. Но она создаёт отдельный `go.Figure`, который затем разбирается по trace'ам и добавляется в другой subplot. Это неэффективно — можно создавать traces напрямую.

### 3.4 libmatrix.py

Полностью экспериментальный модуль. Содержит:
- `compute_feature_matrix_numeric` — вычисление матрицы характеристик 3×6
- `FeatureMatrixDictionary` — словарь уникальных матриц
- `analyze_wave_pairs_numeric` — анализ пар волн

**Не импортируется нигде** в `app_next.py`, `libdatas.py` или `libgraphs.py`. Имеет `if __name__ == "__main__"` блок с демо-данными.

---

## 4. Экспериментальные / неиспользуемые функции

Следующие функции **не вызываются** в текущем рабочем потоке приложения (app_next → libdatas → libgraphs):

| Функция | Файл | Назначение |
|---------|------|------------|
| `calcIntentVector` | libdatas.py | Расчёт вектора намерений (закомментирована в вызывающем коде) |
| `calcInverseMatrix` | libdatas.py | Расчёт инверсной матрицы (вызывается только при `calc_f=True`, но флаг всегда `False`) |
| `reshapeMatrix` | libdatas.py | Изменение размера матрицы (не вызывается) |
| `searchSigns` | libdatas.py | Поиск точек пересечения w1×w2 (закомментирована в вызывающем коде) |
| `draw3D` | libgraphs.py | 3D-поверхность (вызывается только из отдельного callback `view3dSurface`, не часть основного UI) |
| `drawDeepLevelWaves` | libgraphs.py | Отдельный график волн (вызывается только через `createCombinedSubplot`, не самостоятельно) |
| `drawMatrix` | libgraphs.py | Отдельная матрица (вызывается только через `createCombinedSubplot`) |
| `compute_feature_matrix_numeric` | libmatrix.py | Матрица характеристик волн (модуль не импортируется) |
| `FeatureMatrixDictionary` | libmatrix.py | Словарь матриц (модуль не импортируется) |
| `analyze_wave_pairs_numeric` | libmatrix.py | Анализ пар волн (модуль не импортируется) |
| `recalcAllDatasForReverseMatrix` | app_next.py | Callback для reverse matrix (пустой, только `print`) |
| `matchDiffDeep` | libdatas.py | Выбор глубины по режиму (логическая ошибка: ветка `deep_mode==1` продублирована) |

---

## 5. Зависимости

### 5.1 Избыточные зависимости

- `janus` — async queue library, не используется в коде проекта
- `comm` — IPython communication, не используется напрямую
- `retrying` — не импортируется ни в одном файле
- `setuptools`, `zipp`, `importlib_metadata`, `typing_extensions`, `typing-inspection` — транзитивные зависимости, не требуются напрямую

### 5.1 Версии

Все зависимости зафиксированы с точными версиями — хорошо для воспроизводимости.

---

## 6. Безопасность

### 6.1 Base64 декодирование без валидации

```python
content_type, content_string = str(content).split(',', 1)
decoded = base64.b64decode(content_string)
```
Нет проверки размера файла, типа содержимого. Пользователь может загрузить произвольный CSV.

### 6.2 `debug=True` в production

```python
if __name__ == "__main__":
    app.run(debug=True)
```
**Критично для production.** Dash debug mode включает reloader и отладчик Werkzeug, что является вектором атак.

---

## 7. Производительность

### 7.1 Полный перерасчёт при каждом изменении

Каждый клик на `id-but-recalc` вызывает `ld.calcAllDatas()`, которая:
1. Считает все треки (`calcAllTracks`)
2. Считает паспорт (`calcPassport`)
3. Извлекает точки треков (`extract_all_tracks`)
4. Считает точки переходов (`calcLeapPoints`)
5. Считает волны (`calcWavesForDeepLevels`)
6. Считает FFT (`prepare_wave_fft_data`)
7. Считает шансы (`matchPrognosis`)
8. Рисует графики

Время расчёта: ~0.5–2 сек (судя по логам). При больших файлах будет заметно.

### 7.2 Копирование массивов

В `view3dSurface`:
```python
data = np.array(levels['diff_waves'].copy())
```
Избыточное `.copy()` — `np.array()` уже создаёт копию.

### 7.3 `np.rot90` + reverse

В нескольких местах:
```python
matrix_h = np.rot90(matrix_h, k=1)
matrix_h = matrix_h[::-1]
```
Можно заменить одним `np.rot90(matrix_h, k=-1)` или `matrix_h.T[::-1]`.

---

## 8. Документация

- **README.md** — содержит screenshot и краткое описание, но нет инструкций по запуску
- **Отсутствуют docstrings** в большинстве функций
- **Нет type hints** — затрудняет понимание ожидаемых типов аргументов
- **Нет .gitignore исключений** для `.venv/`, `__pycache__/` (хотя директории игнорируются)

---

## 9. Сводка проблем по приоритету

| Приоритет | Проблема | Файл |
|-----------|----------|------|
| **Критический** | `debug=True` в production | app_next.py |
| **Критический** | Нет валидации загружаемых файлов | app_next.py, libdatas.py |
| **Высокий** | Пустой callback без Output | app_next.py:464 |
| **Высокий** | Логическая ошибка в `matchDiffDeep` | libdatas.py:300 |
| **Высокий** | Два источника истины для параметров | app_next.py |
| **Средний** | Dead code / неиспользуемые функции | libdatas.py, app_next.py |
| **Средний** | Избыточные зависимости | requirements.txt |
| **Средний** | Нет type hints и docstrings | все файлы |
| **Низкий** | Drawer никогда не закрывается | app_next.py |
| **Низкий** | Неэффективный rot90+reverse | libdatas.py |

---

## 10. Рекомендации

1. **Убрать `debug=True`** или вынести в переменную окружения
2. **Удалить или пометить** экспериментальные функции (см. раздел 4)
3. **Вынести константы** в отдельный config-файл
4. **Добавить валидацию** входных CSV файлов (размер, формат)
5. **Добавить type hints** как минимум в публичные API функции
6. **Реализовать или удалить** callback `recalcAllDatasForReverseMatrix`
7. **Исправить `matchDiffDeep`** — вторая ветка `elif deep_mode == 1` никогда не выполняется
8. **Убрать `janus`, `retrying`** из requirements.txt если не используются
9. **Добавить механизм закрытия** foot drawer
10. **Рассмотреть инкрементальный перерасчёт** вместо полного при каждом изменении

---

## 11. Оптимизации (выполнены 2026-09-06)

Все изменения применяют новые возможности Dash 4.4, Plotly 7.0 и DMC 2.8.

### 11.1 libdatas.py

| Изменение | Было | Стало |
|-----------|------|-------|
| **Исправлен баг `matchDiffDeep`** | `elif deep_mode == 1 and deep > 0` — недостижимая ветка | Корректная логика: `if deep_mode == 1 and deep > 0: return deep - 1; return deep` |
| **Удалён dead code** | `calcIntentVector` (не вызывается), закомментированные блоки в `calcLeapPoints` | Функция удалена, комментарии убраны |
| **Консолидация загрузки CSV** | `uploadFileByName` и `readFileContent` — дублирование логики | Общая приватная функция `_load_csv_as_array(source)` |
| **Упрощение `prepare_wave_fft_data`** | `wave2 = np.repeat(levels, 2); wave_ds = wave2[::2,:]` — избыточно | `masks = np.sign(levels)` — прямой путь |
| **Type hints** | Отсутствовали | `from __future__ import annotations` + type hints на всех функциях |
| **Docstrings** | Отсутствовали | Добавлены на все публичные функции |

### 11.2 libgraphs.py

| Изменение | Было | Стало |
|-----------|------|-------|
| **Plotly 7 `hovermode`** | Отсутствовал | `hovermode="x unified"` и `hovermode="closest"` на графиках |
| **Plotly 7 `clickmode`** | Только в layout | `clickmode="event+select"` на всех figure |
| **Type hints** | Отсутствовали | Полные type hints с `NDArray[np.int_]`, `NDArray[np.floating]` |
| **Параметр `draw3D`** | `matrix_h` (misleading) | `data` (корректное имя) |
| **Убран дублирующий `update_layout`** | Несколько вызовов подряд | Consolidated в один `fig.update_layout()` |
| **Переиспользование axis_cfg** | Повторяющиеся параметры осей | Общий dict `axis_cfg` для всех subplot |

### 11.3 app_next.py

| Изменение | Было | Стало |
|-----------|------|-------|
| **`debug=True`** | Hardcoded `app.run(debug=True)` | `os.getenv("APP_DEBUG", "true")` — управление через ENV |
| **Пустой callback** | `recalcAllDatasForReverseMatrix` — только `print()` | Удалён, `calc_f` интегрирован в `recalcMatrix` |
| **Drawer toggle** | Всегда возвращал `True` | `toggleFootDrawer` с `State('opened')` — корректное переключение |
| **Глобальные переменные** | `order=0, hist_back=1, ...` — дублирование state | `_STARTUP_*` константы с пояснением «only for initial layout» |
| **Filename bug** | `file_props = {'filename': file_name}` — hardcoded | `file_props = {'filename': filename}` — использует uploaded имя |
| **n_clicks=0** | На menu items в `diffModeMenu` | Удалено — не нужно в Dash 4 |
| **scanDeep max** | Отсутствовал `max` | Добавлен `max=20` |
| **Null-check в callbacks** | `clickData['points'][0]` без проверки | `if not clickData or "points" not in clickData: return no_update` |
| **`.copy()` в view3d** | `np.array(levels['diff_waves'].copy())` | `np.array(levels['diff_waves'])` |
| **Стиль кода** | Смесь одинарных/двойных кавычек, пробелы | PEP 8: двойные кавычки, consistent spacing |

### 11.4 requirements.txt

| Изменение | Было | Стало |
|-----------|------|-------|
| **Зависимости** | 34 пакета (включая транзитивные) | 7 прямых зависимостей: dash, dash-iconify, dash_mantine_components, numpy, pandas, plotly, scipy |
| **Удалены** | `janus`, `retrying`, `comm`, `setuptools`, `zipp`, `importlib_metadata`, `typing_extensions`, `typing-inspection` + 25 транзитивных | Только то, что реально импортируется |

### 11.5 Статистика изменений

```
 app_next.py      | +180/-320  ( чище, меньше дублирования)
 libdatas.py      | +291/-671  (удалён dead code, добавлены type hints)
 libgraphs.py     | +130/-300  (Plotly 7 API, type hints)
 requirements.txt |  -27       (34 → 7 пакетов)
```
