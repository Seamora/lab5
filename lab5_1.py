import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

INIT_PARAMS = {
    'amplitude': 1.0,
    'frequency': 1.0,
    'phase': 0.0,
    'noise_mean': 0.0,
    'noise_covariance': 0.1,
    'show_noise': True,
    'filter_enabled': True,
    'cutoff_freq': 3.0 
}

x = np.linspace(0, 2 * np.pi, 1000)
current_noise = None

def harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise):
    global current_noise
    y_clean = amplitude * np.sin(frequency * x + phase)
    if show_noise:
        if current_noise is None:
            current_noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), size=x.shape)
        return y_clean + current_noise
    else:
        return y_clean

def apply_filter(y, cutoff_freq):
    b, a = butter(N=6, Wn=cutoff_freq / (0.5 * len(x)), btype='low')
    y_filtered = filtfilt(b, a, y)
    return y_filtered

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.45)

y1 = harmonic_with_noise(
    INIT_PARAMS['amplitude'],
    INIT_PARAMS['frequency'],
    INIT_PARAMS['phase'],
    INIT_PARAMS['noise_mean'],
    INIT_PARAMS['noise_covariance'],
    INIT_PARAMS['show_noise']
)

line1, = ax1.plot(x, y1, label='Гармоніка з шумом')
ax1.set_title("Завдання 1: Гармоніка з шумом")
ax1.set_ylim(-5, 5)

y2 = harmonic_with_noise(
    INIT_PARAMS['amplitude'],
    INIT_PARAMS['frequency'],
    INIT_PARAMS['phase'],
    INIT_PARAMS['noise_mean'],
    INIT_PARAMS['noise_covariance'],
    INIT_PARAMS['show_noise']
)

y2_filtered = apply_filter(y2, INIT_PARAMS['cutoff_freq'])
line2, = ax2.plot(x, y2_filtered, label='Відфільтрований сигнал', color='orange')
ax2.set_title("Завдання 2: Відфільтрована гармоніка")
ax2.set_ylim(-5, 5)

ax_amp = plt.axes([0.25, 0.35, 0.65, 0.03])
slider_amp = Slider(ax_amp, 'Амплітуда', 0.1, 5.0, valinit=INIT_PARAMS['amplitude'])

ax_freq = plt.axes([0.25, 0.30, 0.65, 0.03])
slider_freq = Slider(ax_freq, 'Частота', 0.1, 10.0, valinit=INIT_PARAMS['frequency'])

ax_phase = plt.axes([0.25, 0.25, 0.65, 0.03])
slider_phase = Slider(ax_phase, 'Фаза', 0.0, 2*np.pi, valinit=INIT_PARAMS['phase'])

ax_noise_mean = plt.axes([0.25, 0.20, 0.65, 0.03])
slider_noise_mean = Slider(ax_noise_mean, 'Середнє шуму', -1.0, 1.0, valinit=INIT_PARAMS['noise_mean'])

ax_noise_cov = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_noise_cov = Slider(ax_noise_cov, 'Дисперсія шуму', 0.01, 1.0, valinit=INIT_PARAMS['noise_covariance'])

ax_cutoff = plt.axes([0.25, 0.10, 0.65, 0.03])
slider_cutoff = Slider(ax_cutoff, 'Частота зрізу фільтру', 0.1, 10.0, valinit=INIT_PARAMS['cutoff_freq'])

ax_checkbox1 = plt.axes([0.025, 0.4, 0.15, 0.15])
checkbox = CheckButtons(ax_checkbox1, ['Шум'], [INIT_PARAMS['show_noise']])

ax_checkbox2 = plt.axes([0.025, 0.25, 0.15, 0.15])
filter_checkbox = CheckButtons(ax_checkbox2, ['Фільтр'], [INIT_PARAMS['filter_enabled']])

ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_button, 'Reset')

def update(val):
    global current_noise
    if val == 'noise':
        current_noise = np.random.normal(slider_noise_mean.val, np.sqrt(slider_noise_cov.val), size=x.shape)

    y = harmonic_with_noise(
        slider_amp.val,
        slider_freq.val,
        slider_phase.val,
        slider_noise_mean.val,
        slider_noise_cov.val,
        checkbox.get_status()[0]
    )
    line1.set_ydata(y)

    if filter_checkbox.get_status()[0]:
        y_filtered = apply_filter(y, slider_cutoff.val)
        line2.set_ydata(y_filtered)
    else:
        line2.set_ydata(y)

    fig.canvas.draw_idle()

def reset(event):
    global current_noise
    current_noise = None
    slider_amp.reset()
    slider_freq.reset()
    slider_phase.reset()
    slider_noise_mean.reset()
    slider_noise_cov.reset()
    slider_cutoff.reset()
    checkbox.set_active(0 if not INIT_PARAMS['show_noise'] else 1)
    filter_checkbox.set_active(0 if not INIT_PARAMS['filter_enabled'] else 1)
    update('noise')

slider_amp.on_changed(lambda _: update('signal'))
slider_freq.on_changed(lambda _: update('signal'))
slider_phase.on_changed(lambda _: update('signal'))
slider_noise_mean.on_changed(lambda _: update('noise'))
slider_noise_cov.on_changed(lambda _: update('noise'))
slider_cutoff.on_changed(lambda _: update('signal'))
checkbox.on_clicked(lambda _: update('signal'))
filter_checkbox.on_clicked(lambda _: update('signal'))
button.on_clicked(reset)

plt.show()
