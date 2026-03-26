import sys
import subprocess
import importlib


# --- Dependency installer with correct import names ---
REQUIRED_PACKAGES = [
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('plotly', 'plotly'),
    ('scikit-learn', 'sklearn'),
    ('scipy', 'scipy'),
]

def install_and_import(pip_name, import_name):
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
    finally:
        globals()[import_name] = importlib.import_module(import_name)

for pip_name, import_name in REQUIRED_PACKAGES:
    install_and_import(pip_name, import_name)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import lognorm
import matplotlib as mpl

# --- Parameters ---
N_PATHS = 1200
N_DAYS = 252
S0 = 1.0
MU = 0.0
SIGMA = 0.25
SEED = 42
np.random.seed(SEED)

# --- Simulate GBM Paths ---
def simulate_gbm_paths(n_paths, n_days, s0, mu, sigma):
    dt = 1/n_days
    increments = np.random.normal((mu-0.5*sigma**2)*dt, sigma*np.sqrt(dt), (n_paths, n_days))
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths,1)), log_paths])
    paths = s0 * np.exp(log_paths)
    return paths

paths = simulate_gbm_paths(N_PATHS, N_DAYS, S0, MU, SIGMA)
final_vals = paths[:, -1]
mean_path = np.mean(paths, axis=0)

# --- Color by outcome (blue to red) ---
sort_idx = np.argsort(final_vals)
norm = mpl.colors.Normalize(vmin=np.min(final_vals), vmax=np.max(final_vals))
cmap = mpl.cm.get_cmap('coolwarm')
colors = cmap(norm(final_vals))

# --- PDF for final values (lognormal) ---
shape = SIGMA * np.sqrt(1)
scale = S0 * np.exp((MU-0.5*SIGMA**2)*1)
x_pdf = np.linspace(np.min(final_vals), np.max(final_vals), 200)
pdf = lognorm.pdf(x_pdf, s=shape, scale=scale)

# --- Expected value line ---
expected_val = S0 * np.exp(MU*1)

# --- Plot ---
plt.style.use('seaborn-v0_8-white')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'


import matplotlib.animation as animation

fig = plt.figure(figsize=(10,7))
gs = gridspec.GridSpec(1, 2, width_ratios=[4,1], wspace=0.05)
ax_paths = plt.subplot(gs[0])
ax_hist = plt.subplot(gs[1], sharey=ax_paths)

# --- Prepare lines for animation ---
time_grid = np.linspace(0,1,N_DAYS+1)
lines = [ax_paths.plot([], [], color=colors[i], lw=1, alpha=0.5, zorder=1)[0] for i in range(N_PATHS)]
mean_line, = ax_paths.plot([], [], color='k', lw=2, linestyle='--', label=r'$E[X_t]$', zorder=3)
cone_fill = None  # Will be drawn in animate
exp_line = ax_paths.axhline(expected_val, color='deepskyblue', ls='--', lw=2, label=r'$E[X_T]$', zorder=4)

# --- Static histogram and PDF ---
hist_vals, bins, patches = ax_hist.hist(final_vals, bins=30, orientation='horizontal', density=True,
                                        color='orange', alpha=0.5, label=r'$X_T$ pdf', zorder=1)
ax_hist.plot(pdf, x_pdf, color='orange', lw=2, label=r'$X_T$ pdf', zorder=2)
ax_hist.axhline(expected_val, color='deepskyblue', ls='--', lw=2, label=r'$E[X_T]$', zorder=3)
ax_hist.set_xlabel('')
ax_hist.set_ylabel('')
ax_hist.set_xticks([])
ax_hist.set_yticks([])
ax_hist.legend(fontsize=12, loc='upper right')
plt.setp(ax_hist.get_yticklabels(), visible=False)

# --- Labels and style ---
ax_paths.set_xlabel(r'$t$', fontsize=16)
ax_paths.set_ylabel(r'$X(t)$', fontsize=16)
ax_paths.set_title('Geometric Brownian Motion\nSimulated Paths $X_t, t \in [t_0, T]$', fontsize=16)
ax_paths.grid(True, alpha=0.3)
ax_paths.legend(fontsize=12, loc='upper left')

# --- Probability cone arrays ---
log_mean = np.log(S0) + (MU - 0.5*SIGMA**2) * time_grid
log_std = SIGMA * np.sqrt(time_grid)
cone_upper = np.exp(log_mean + log_std)
cone_lower = np.exp(log_mean - log_std)

# --- Animation functions ---
def init():
    for line in lines:
        line.set_data([], [])
    mean_line.set_data([], [])
    # Remove any previous cone fills
    [c.remove() for c in ax_paths.collections]
    return lines + [mean_line]

def animate(frame):
    t = frame
    for i, line in enumerate(lines):
        line.set_data(time_grid[:t+1], paths[i, :t+1])
    mean_line.set_data(time_grid[:t+1], mean_path[:t+1])
    # Remove previous cone fills
    [c.remove() for c in ax_paths.collections]
    ax_paths.fill_between(time_grid[:t+1], cone_lower[:t+1], cone_upper[:t+1], color='gray', alpha=0.2, zorder=2)
    return lines + [mean_line]

ani = animation.FuncAnimation(fig, animate, frames=N_DAYS+1, init_func=init,
                              interval=20, blit=True, repeat=False)
plt.tight_layout()
plt.show()
