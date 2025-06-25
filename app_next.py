import numpy as np
from dash import Dash, dcc, Input, Output, State, callback, no_update, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify

import libdatas as ld
import libgraphs as lg

#Константы по умолчанию на старте:
order=0
hist_back=1
matrix_size=200
deep_mode=0
deep=0
sorted=False
dm_icon0=DashIconify(icon="streamline:wave-signal-solid")
dm_icon1=DashIconify(icon="material-symbols:airwave-rounded")
dm_icon2=DashIconify(icon="tabler:waves-electricity")
dm_icons_pack =[dm_icon0, dm_icon1, dm_icon2]

file_name = '4x20.csv'
file_data, matrix_data, matrix_size, max_size, max_order, max_nums, track_num = ld.uploadFileByName(
    file_name,
    order,
    matrix_size,
    hist_back,
    sorted
    )

fig_matrix, fig_chances, levels_fft_store, mr_table, calc_time = ld.calcAllDatas(
    matrix_data[:, order], matrix_size, max_nums, track_num, deep_mode)

data_props = {'filename': file_name, 'nums': max_nums, 'orders': max_order, 'file_data': file_data, 'deep_mode': deep_mode, 'deep': deep}

#==================LAYOUT=====================

app = Dash(external_stylesheets=dmc.styles.ALL, title="Prognosis Ai.Next", prevent_initial_callbacks=True)  #error!:, show_undo_redo=True

#===============HEADER COMPONENTS=============:

def fileInfoCard(filename, arr_size, max_cols, max_num, calc_time):
    return dmc.HoverCard(
        width=240,
        withArrow=True,
        position='bottom-start',
        offset=10,
        children=[
            dmc.HoverCardTarget(
                dcc.Upload(
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:file-type-csv"), id="id-load-button"
                    ),
                    id='id-upload'),
            ),
            dmc.HoverCardDropdown(
                [
                    dmc.List(
                        [
                            dmc.ListItem([dmc.Text(f"File name: {filename}")]),
                            dmc.ListItem([dmc.Text(f"Matrix max size: {max_size}")]),
                            dmc.ListItem([dmc.Text(f"Orders: {max_cols}")]),
                            dmc.ListItem([dmc.Text(f"Max tracks: {max_num}")]),
                            dmc.ListItem([dmc.Text(f"Time calc: {calc_time:.3f} sec", c='red')],
                                        icon=DashIconify(icon="mingcute:time-duration-fill", width=50)),
                        ],
                    ),
                ],
            ),
        ],
    )
#------------------------------------------------------------------------

orderSelector = dmc.Select(
    data=[{"label": f"Order {i+1}", "value": str(i+1)} for i in range(max_order)],
    value=str(order+1),
    size='xs',
    leftSection=DashIconify(icon="icon-park-solid:horizontal-tidy-up"),
    w=100,
    allowDeselect=False,
    id="id-order-select",
    #persistence=True
)
#------------------------------------------------------------------------

switchSort = dmc.Switch(
    size="xs",
    #offLabel=DashIconify(icon="iconamoon:arrow-right-1-fill", width=20),
    onLabel=DashIconify(icon="tabler:sort-0-9", width=20),
    radius="md",
    label="Sorted",
    labelPosition="left",
    checked=False,
    id="id-switch-sort"
)
#------------------------------------------------------------------------

matrixSize = dmc.NumberInput(
    value=matrix_size,
    min=2,
    max=max_size,
    prefix='M-size: ',
    leftSection=DashIconify(icon="healthicons:ui-menu-grid"),
    w=150,
    size='xs',
    id="id-matrix-size"
)
#------------------------------------------------------------------------

historyBack = dmc.NumberInput(
    id="id-hist-back",
    min=0,
    value=1,
    prefix='H-back: ',
    leftSection=DashIconify(icon="ph:arrow-fat-lines-right-bold"),
    w=120,
    size='xs',
)
#------------------------------------------------------------------------

scanDeep = dmc.NumberInput(
    id="id-markov-deep",
    min=2,
    value=2,
    prefix='M-deep: ',
    leftSection=DashIconify(icon="gravity-ui:list-timeline"),
    w=120,
    size='xs',
)
#------------------------------------------------------------------------

switchCalkMatrixF = dmc.Switch(
    size="xs",
    #offLabel=DashIconify(icon="iconamoon:arrow-right-1-fill", width=20),
    onLabel=DashIconify(icon="fa6-solid:left-right", width=20),
    radius="md",
    label="Calc Matrix-F",
    labelPosition="left",
    checked=False,
    id="id-switch-calc-matrix-f"
)
#------------------------------------------------------------------------

trackSelector=dmc.Select(
    #минимум для track_num == 1:
    data=ld.setTrackList(track_num, max_nums, hist_back),
    value=str(track_num+1), #во входном файле числа с 1. При индексах треков win_num-1
    size='xs',
    w=130,
    leftSection=DashIconify(icon="bi:bar-chart-steps"),
    allowDeselect=False,
    id="id-track-select",
    persistence=True
)
#------------------------------------------------------------------------

diffModeMenu = dmc.Menu(
    children=[
        dmc.MenuTarget(
            dmc.ActionIcon(dm_icons_pack[deep_mode], id="id-but-diff-mode"),
            boxWrapperProps="children"
        ),
        dmc.MenuDropdown(
            [
                dmc.MenuLabel("Deep shift mode:"),
                dmc.MenuDivider(),
                dmc.MenuItem("Stazis", id="id-mode-stazis", n_clicks=0,
                    leftSection=dm_icons_pack[0]
                ),
                dmc.MenuItem("Deep Y", id="id-mode-deep", n_clicks=0,
                    leftSection=dm_icons_pack[1]
                ),
                dmc.MenuItem("Smart mode", id="id-mode-smart", n_clicks=0,
                    leftSection=dm_icons_pack[2],
                    color="yellow"
                ),
            ]
        ),
    ],
    id = "id-diff-mode",
    trigger = "hover",
    withArrow = True,
)
#------------------------------------------------------------------------

but_recalc = dmc.ActionIcon(
    DashIconify(icon="fluent:calculator-arrow-clockwise-20-regular"),
    #variant="outline",
    color="green",
    variant="filled",
    n_clicks=0,
    id="id-but-recalc",
)
#------------------------------------------------------------------------


but_view_3d = dmc.ActionIcon(
    DashIconify(icon="mage:box-3d-scan"),
    #variant="outline",
    color="green",
    variant="filled",
    id="id-but-view-3d",
)
#------------------------------------------------------------------------

but_view_foot_drawer = dmc.ActionIcon(
    DashIconify(icon="majesticons:table-line"),
    id='id-but-view-foot-drawer'
)
#------------------------------------------------------------------------

app_header = dmc.AppShellHeader([
    dcc.Store(id='id-file-props-store', data=data_props),
    dmc.Grid([
        dmc.GridCol(
            [
                dmc.Group([
                    dmc.Text("XaosLab", size="xl", fw='bolder', variant="gradient",
                        gradient={"from": "indigo", "to": "yellow", "deg": 45},),
                    dmc.Box([
                            fileInfoCard(file_name, matrix_size, max_order, max_nums, calc_time)], id = "id-info-card"),
                            switchSort,
                            orderSelector,
                            historyBack,
                            scanDeep,
                            matrixSize,
                            trackSelector,
                            diffModeMenu,
                            switchCalkMatrixF,
                            but_recalc,
                        ]
                    ),
            ],
            span="auto"
        ),
        dmc.GridCol([
            dmc.Group([but_view_foot_drawer, but_view_3d], justify='flex-end')],
            span="content", offset=2
        )],
    )],
    mx=10
)
#------------------------------------------------------------------------

excluderMenu = dmc.Menu(
    [
            dmc.MenuTarget(
                dmc.ActionIcon(
                    DashIconify(icon="carbon:skill-level-intermediate", rotate=1, flip="horizontal"),
                    )
                ),
            dmc.MenuDropdown(
                [
                    dmc.MenuLabel("Exclude:"),
                    dmc.MenuItem(
                        "Selected", leftSection=DashIconify(icon="tabler:settings")
                    ),
                    dmc.MenuItem(
                        "Unselected", leftSection=DashIconify(icon="tabler:message")
                    ),
                    dmc.MenuItem("Save", leftSection=DashIconify(icon="tabler:photo")),
                    dmc.MenuDivider(),
                    dmc.MenuLabel("Danger Zone"),
                    dmc.MenuItem(
                        "Restore last",
                        leftSection=DashIconify(icon="tabler:arrows-left-right"),
                    ),
                    dmc.MenuItem(
                        "Restore all",
                        leftSection=DashIconify(icon="tabler:trash"),
                        color="red",
                    ),
                ]
            ),
    ],
    trigger="hover",
)
#------------------------------------------------------------------------
mr_table = dmc.Table(
    id='id-mr-table',
    striped=True,
    highlightOnHover=True,
    withTableBorder=True,
    withColumnBorders=True,
    data = {"head": mr_table.columns.tolist(), "body": mr_table.values.tolist()},
    verticalSpacing=1,
    horizontalSpacing=2,
    fz=10,
)

foot_drawer = dmc.Drawer(
    [
        mr_table,
    ],
    title='Relations Markers',
    position='bottom',
    size='60%',
    padding='md',
    id='id-foot-drawer'
)
#------------------------------------------------------------------------

app_main=dmc.AppShellMain(
    [
        dcc.Store(id="id-matrix-store", data=fig_matrix),
        dcc.Store(id="id-levels-store", data=levels_fft_store),
        #dcc.Store(id="id-clicks-data", data=clicks_layer)
        dmc.Grid(
            [
                dmc.GridCol(
                    [dcc.Slider(id="id-slider-balance", min=0.1, step=0.05, value=0.5, max = 1.0, vertical=True, verticalHeight = 900+20, updatemode='mouseup', persistence=True)],
                    span=1,
                    mt=0
                ),
                dmc.GridCol(
                    [dcc.Graph(figure=fig_matrix, id="id-matrix")],
                span=44,
                ),
                dmc.GridCol(
                    [
                        dmc.Stack(
                            [
                                dmc.Text("Prognosis.Next:", fw='bolder'),
                                excluderMenu,
                                dcc.Graph(figure=fig_chances, id="id-chances"), 
                            ],
                            align="left",
                            gap="5",
                        )
                    ],
                    span=5,
                ),
            ],
            columns=50,
            overflow='scroll',
            gutter=5,
            ml=5,
            mt=45,
        ),
    foot_drawer,
    ],
)
#------------------------------------------------------------------------

layout = dmc.AppShell(
    [
        app_header,
        app_main,
    ],
    header={"height": "lg"},
)
#------------------------------------------------------------------------

app.layout = dmc.MantineProvider(
    [
        layout,
    ],
    forceColorScheme="dark",
)
#------------------------------------------------------------------------

#====================================CALLBACKS=============================

@callback(
    # Установка компонентов по умолчанию для загруженного файла архива:
    Output('id-matrix', 'figure', allow_duplicate=True),
    Output('id-chances', 'figure', allow_duplicate=True),
    Output('id-levels-store', 'data', allow_duplicate=True),
    Output('id-file-props-store', 'data', allow_duplicate=True),
    Output('id-hist-back', 'value'),
    Output('id-matrix-size', 'max'),
    Output('id-matrix-size', 'value'),
    Output('id-order-select', 'data'),
    Output('id-order-select', 'value'),
    Output('id-track-select', 'data'),
    Output('id-track-select', 'value'),
    Output('id-info-card', 'children'),
    Input('id-upload', 'contents'),
    State('id-switch-sort', 'checked'),
    State('id-upload', 'filename'),
    prevent_initial_call=True,
)
def uploadFile(content, sorted, filename):  #
    csv_arr, matrix_max, max_order = ld.readFileContent(content, sorted)
    print('max',matrix_max)
    order = 0
    orders_list = [{"label": f"Order {i+1}", "value": str(i+1)} for i in range(max_order)]
    h_back = 1
    deep = 0

    max_nums = max(csv_arr[:, order])
    tracks_list = [{"label": f"Track {i+1}", "value": str(i+1)} for i in range(max_nums)]
    
    track_num = ld.autoSelectNextTrack(csv_arr, h_back, order)

    matrix_size = 250
    if matrix_size > matrix_max:
        matrix_size = matrix_max - hist_back
    matrix_graf, chances_bar, levels_store, mr_table, calc_time = ld.calcAllDatas(csv_arr[hist_back:matrix_size+hist_back, order], matrix_size, max_nums, track_num, deep_mode=0, subplots_balance=[0.5,0.5], calc_f=False)
    
    file_props = {'filename': file_name, 'nums': max_nums, 'orders': max_order, 'file_data': csv_arr, 'deep_mode': deep_mode, 'deep': deep}
    file_info_card = fileInfoCard(filename, matrix_max, max_order, max_nums, calc_time)
    return matrix_graf, chances_bar, levels_store, file_props, h_back, matrix_max, matrix_size, orders_list, str(order+1), tracks_list, str(track_num+1), file_info_card
#---------------------------------------------

#Выбор режима анализа взаимосвязей волн событийной глубины и матрицы:
@callback(
    Output('id-file-props-store', 'data', allow_duplicate=True),
    Output('id-but-diff-mode', 'children'),
    Input('id-mode-stazis', 'n_clicks'),
    Input('id-mode-deep', 'n_clicks'),
    Input('id-mode-smart', 'n_clicks'),
    State('id-file-props-store', 'data'),
    prevent_initial_call=True,
)
def setDiffMode(mode0, mode1, mode2, file_props):
    deep_mode = 0
    if ctx.triggered_id == "id-mode-stazis":
        deep_mode = 0
    elif ctx.triggered_id == "id-mode-deep":
        deep_mode = 1
    elif ctx.triggered_id == "id-mode-smart":
        deep_mode = 2
    
    file_props['deep_mode'] = deep_mode
    but_icon = dmc.ActionIcon(dm_icons_pack[deep_mode])
    return file_props, but_icon
#---------------------------------------------    

# Перерасчет матрицы и всех графиков при смене ордера, размера матрицы, режима сорт и пр.
@callback(
    Output('id-matrix', 'figure', allow_duplicate=True),
    Output('id-chances', 'figure', allow_duplicate=True),
    Output('id-levels-store', 'data', allow_duplicate=True),
    Output('id-track-select', 'value', allow_duplicate=True),
    Input('id-but-recalc', 'n_clicks'),
    Input('id-order-select','value'),
    Input('id-hist-back','value'),
    Input('id-matrix-size', 'n_blur'),
    Input('id-slider-balance', 'value'),
    State('id-matrix-size', 'value'),
    State('id-file-props-store', 'data'),
    State('id-track-select', 'value'),
    prevent_initial_call=True,
)
def recalcMatrix(_, sel_order, hist_back, n_blur, slider_val, m_size, file_props, sel_track):
    order = int(sel_order)-1
    file_data = np.array(file_props['file_data'])
    order_data = file_data[hist_back:m_size+hist_back, order]
    max_nums = file_props['nums']
    deep_mode = file_props['deep_mode']


    # Если смена ордера или hist_back > 0, то автосмена номера трека
    # или track_num остается выбранный:
    track_num = int(sel_track) - 1
    if ctx.triggered_id == 'id-order-select' or (ctx.triggered_id == 'id-hist-back' and hist_back > 0):
        track_num = ld.autoSelectNextTrack(file_data, hist_back, order)
    track_val = str(track_num + 1)

    # Пропорциональный перерасчет масштаба высоты графиков:
    subplots_balance = [1.0 - slider_val, slider_val]

    matrix_fig, chances_fig, levels_fft_store, mr_table, time_recalc = ld.calcAllDatas(order_data, int(m_size), int(max_nums), track_num, deep_mode, subplots_balance)
    print('time recalc', time_recalc)
    return matrix_fig, chances_fig, levels_fft_store, track_val
#---------------------------------------------

#Обновление графиков для выбранной по клику глубины матрицы:
@callback(
    Output('id-matrix', 'figure', allow_duplicate=True),
    Input('id-matrix', 'clickData'),
    State('id-matrix', 'figure'),
    State('id-track-select', 'value'),
    State('id-levels-store', 'data'),
    State('id-file-props-store', 'data'),
    prevent_initial_call=True,
)
def updateLevelOnClickMatrix(clickData, matrix, track_num, levels_fft_store, file_props): #, matrix_fig, track_num)
    # извлекаем координаты матрицы x, y, z
    point = clickData['points'][0]
    
    #x = int(point['x'])
    y = int(point['y']) #for select deep level's plot
    idx_subplot = point['curveNumber']
    #print(f"x: {x}, y: {y}, subfig: {idx_subplot}")  #, z: {z}

    if idx_subplot > 3:
        #num = str(int(track_num)-1)
        deep = str(y)
      
        #Запоминаем shape для линии глубины уровня:
        idx_last_shape = len(matrix['layout']['shapes']) - 1
        deep_shape = matrix['layout']['shapes'][idx_last_shape]

        # Обновление fft & wave для выбранного событийного уровня:
        levels_fft = levels_fft_store['levels_fft']
        level = levels_fft[deep]
        fft_y = level['fft_y']

        # Оптимизируем вычисление y_min и y_max для корректного диапазона
        # с учетом deep_mode:
        deep_mode = file_props['deep_mode']
        diffs_w1 = np.array(levels_fft_store['diff_waves'], dtype=int)
        levels_w2 = np.array(levels_fft_store['matrix_levels'], dtype=int)
        y_min, y_max = ld.calcWavesMinMax(diffs_w1, levels_w2, y, deep_mode)

        diff_wave = diffs_w1[:, y]
        matrix_level = levels_w2[:, y]

        # Разделяем positive/negative для Bar subplots:
        matrix['data'][0]['y'] = fft_y[0]   #pos
        matrix['data'][1]['y'] = fft_y[1]   #neg
        matrix['data'][2]['y'] = diff_wave
        matrix['data'][3]['y'] = matrix_level

        #Зоны поиска с аннотациями:
        tr_num = int(track_num)-1
        tr_zones = levels_fft_store['leaps_zones'][tr_num]
        #print('tr_zones onclick', tr_zones)
        
        shapes, annotations = ld.createShapesAndAnnotations(tr_zones, y_min, y_max)
        matrix['layout']['shapes'] = shapes
        matrix['layout']['annotations'] = annotations

        # добавление обновленной Y-deep линии:
        deep_shape['y0'] = y
        deep_shape['y1'] = y
        matrix['layout']['shapes'].append(deep_shape)

        #Граfик анализа точек пересечения w1xw2:
        #w1 = np.array(levels_fft_store['diff_waves'].copy())
        #w2 = np.array(levels_fft_store['matrix_levels'].copy())
        #signs = ld.searchSigns(w1, w2, deep=y)
        #print(signs)

        return matrix
    
    else:
        return no_update
#--------------------------------------------------

@callback(
    Output('id-foot-drawer', 'opened'),
    Input('id-but-view-foot-drawer', 'n_clicks'),
    prevent_initial_call=True,
)
def showFootDrawer(click):
    return True

#--------------------------------------------------
@callback(
    Input('id-but-view-3d', 'n_clicks'),
    State('id-levels-store', 'data'),
    prevent_initial_call=True,
)
def view3dSurface(n_clicks, levels):
    data = np.array(levels['diff_waves'].copy())
    fig_3d = lg.draw3D(data)
    fig_3d.show()
#--------------------------------------------------

@callback(
    Input('id-switch-calc-matrix-f', 'checked')
)
def recalcAllDatasForReverseMatrix(on_off):
    print(on_off)
#---------------------------------------------



if __name__ == "__main__":
    app.run(debug=True)