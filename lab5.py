import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

INIT_PARAMS = {
    'amplitude': 1.0,
    'frequency': 1.0,
    'phase': 0.0,
    'noise_mean': 0.0,
    'noise_covariance': 0.1,
    'show_noise': True,
    'filter_type': 'None'
}

x = np.linspace(0, 2 * np.pi, 1000)
current_noise = np.random.normal(INIT_PARAMS['noise_mean'], np.sqrt(INIT_PARAMS['noise_covariance']), size=x.shape)

def generate_signal(amplitude, frequency, phase):
    return amplitude * np.sin(frequency * x + phase)

def add_noise(signal, noise_mean, noise_covariance, show_noise):
    global current_noise
    if show_noise:
        return signal + current_noise
    return signal

def custom_filter(signal, filter_type):
    if filter_type == 'Moving Average':
        return np.convolve(signal, np.ones(5)/5, mode='same')
    elif filter_type == 'Mean Subtraction':
        return signal - np.mean(signal)
    return signal

app = dash.Dash(__name__)
app.title = "Лабораторна №5"

app.layout = html.Div([
    html.H2("Налаштування графіка "),

    html.Label("Амплітуда"),
    dcc.Slider(
        id='amplitude',
        min=0.1, max=5.0, step=0.1, value=INIT_PARAMS['amplitude'],
        marks={round(i, 1): str(round(i, 1)) for i in np.arange(0.1, 5.1, 0.5)},
        tooltip={"always_visible": False, "placement": "bottom"},
        included=False
    ),

    html.Label("Частота"),
    dcc.Slider(
        id='frequency',
        min=0.1, max=10.0, step=0.1, value=INIT_PARAMS['frequency'],
        marks={round(i, 1): str(round(i, 1)) for i in np.arange(0.1, 10.1, 1)},
        tooltip={"always_visible": False, "placement": "bottom"},
        included=False
    ),

    html.Label("Фаза"),
    dcc.Slider(
        id='phase',
        min=0.0, max=2*np.pi, step=0.1, value=INIT_PARAMS['phase'],
        marks={round(i, 1): str(round(i, 1)) for i in np.arange(0, 2*np.pi+0.1, np.pi/2)},
        tooltip={"always_visible": False, "placement": "bottom"},
        included=False
    ),

    html.Label("Середнє шуму"),
    dcc.Slider(
        id='noise_mean',
        min=-1.0, max=1.0, step=0.1, value=INIT_PARAMS['noise_mean'],
        marks={round(i, 1): str(round(i, 1)) for i in np.arange(-1.0, 1.1, 0.5)},
        tooltip={"always_visible": False, "placement": "bottom"},
        included=False
    ),

    html.Label("Дисперсія шуму"),
    dcc.Slider(
        id='noise_covariance',
        min=0.01, max=1.0, step=0.01, value=INIT_PARAMS['noise_covariance'],
        marks={round(i, 2): str(round(i, 2)) for i in np.arange(0.01, 1.01, 0.25)},
        tooltip={"always_visible": False, "placement": "bottom"},
        included=False
    ),

    html.Label("Фільтр"),
    dcc.Dropdown(
        id='filter_type',
        options=[
            {'label': 'Без фільтрації', 'value': 'None'},
            {'label': 'Віднімання середнього', 'value': 'Mean Subtraction'},
            {'label': 'Ковзне середнє', 'value': 'Moving Average'}
        ],
        value=INIT_PARAMS['filter_type']
    ),

    html.Label("Показати шум"),
    dcc.Checklist(
        id='show_noise',
        options=[{'label': 'Включити', 'value': 'show'}],
        value=['show'] if INIT_PARAMS['show_noise'] else []
    ),

    html.Button('Reset', id='reset', n_clicks=0),
    html.Hr(),

    dcc.Graph(id='original_signal'),
    dcc.Graph(id='noisy_signal'),
    dcc.Graph(id='filtered_signal')
])


@app.callback(
    [Output('original_signal', 'figure'),
     Output('noisy_signal', 'figure'),
     Output('filtered_signal', 'figure')],
    [Input('amplitude', 'value'),
     Input('frequency', 'value'),
     Input('phase', 'value'),
     Input('noise_mean', 'value'),
     Input('noise_covariance', 'value'),
     Input('show_noise', 'value'),
     Input('filter_type', 'value'),
     Input('reset', 'n_clicks')]
)
def update_graph(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise, filter_type, n_clicks):
    global current_noise
    trigger = dash.callback_context.triggered
    if trigger and 'reset' in trigger[0]['prop_id']:
        amplitude = INIT_PARAMS['amplitude']
        frequency = INIT_PARAMS['frequency']
        phase = INIT_PARAMS['phase']
        noise_mean = INIT_PARAMS['noise_mean']
        noise_covariance = INIT_PARAMS['noise_covariance']
        filter_type = INIT_PARAMS['filter_type']
        show_noise = ['show'] if INIT_PARAMS['show_noise'] else []

    current_noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), size=x.shape)

    clean = generate_signal(amplitude, frequency, phase)
    noisy = add_noise(clean, noise_mean, noise_covariance, 'show' in show_noise)
    filtered = custom_filter(noisy, filter_type)

    fig1 = go.Figure(data=[go.Scatter(x=x, y=clean, mode='lines')])
    fig1.update_layout(title='Чиста гармоніка')

    fig2 = go.Figure(data=[go.Scatter(x=x, y=noisy, mode='lines')])
    fig2.update_layout(title='Сигнал з шумом')

    fig3 = go.Figure(data=[go.Scatter(x=x, y=filtered, mode='lines')])
    fig3.update_layout(title='Після фільтрації')

    return fig1, fig2, fig3

if __name__ == '__main__':
    app.run(debug=True)




