# -*- coding: utf-8 -*-
"""Export benchmark charts as PNG files for docs/."""
import os, glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, 'docs', 'assets')
os.makedirs(ASSETS, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, 'data', 'benchmark-results.csv'))
df = df[df['run_label'] == 'local_fastbrain']
def col(name):
    return next(c for c in df.columns if c.lower() == name)

rows = []
for _, r in df.iterrows():
    rows.append(dict(backend=str(r[col('framework')]), t=float(r[col('t_run')]),
                     n=int(r[col('n_run')]), sim=float(r[col('sim_time')])))
rows.sort(key=lambda x: (x['backend'], x['n']))
for r in rows:
    r['rt'] = r['t'] * r['n'] / r['sim']

plt.rcParams.update({'figure.facecolor': '#0a0e1a', 'axes.facecolor': '#0a0e1a',
                     'axes.edgecolor': '#445', 'axes.labelcolor': '#ccd',
                     'xtick.color': '#99a', 'ytick.color': '#99a', 'text.color': '#dde'})

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
names = [f"{r['backend'].replace(' (CUDA Graph)', '').replace(' (CUDA)', '')}\nn={r['n']}" for r in rows]
colors = ['#5b8ff9' if 'FastBrain' in r['backend'] else '#556' for r in rows]
axes[0].bar(range(len(rows)), [r['sim'] for r in rows], color=colors)
axes[0].set_xticks(range(len(rows))); axes[0].set_xticklabels(names, fontsize=7)
axes[0].set_ylabel('sim time (s)'); axes[0].set_title('Simulation time (t_run = 0.5 s, sugar 200 Hz)')
axes[1].bar(range(len(rows)), [r['rt'] for r in rows], color=colors)
axes[1].axhline(1.0, color='#e8684a', ls='--', lw=1, label='real-time (1x)')
axes[1].set_xticks(range(len(rows))); axes[1].set_xticklabels(names, fontsize=7)
axes[1].set_ylabel('real-time ratio'); axes[1].set_title('Speed vs real time'); axes[1].legend(fontsize=8)
fig.savefig(os.path.join(ASSETS, 'chart_speed.png'), dpi=130, bbox_inches='tight')
plt.close(fig)

vcsv = sorted(glob.glob(os.path.join(ROOT, 'experiments', 'out', 'validation', 'validate_matrix_*.csv')))[-1]
vdf = pd.read_csv(vcsv)
drift = vdf[vdf['kind'] == 'drift']
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(range(1, len(drift)+1), drift['rate_corr'], 'o-', color='#5b8ff9', label='fast vs ref, per-1s window')
ax.axhline(0.970, color='#e8684a', ls='--', lw=1, label='seed-noise floor')
ax.set_xlabel('window (1 s each)'); ax.set_ylabel('rate correlation')
ax.set_title('Long-run drift: sugar 10 s, dt = 0.25 ms'); ax.legend(fontsize=8); ax.set_ylim(0.94, 1.0)
fig.savefig(os.path.join(ASSETS, 'chart_drift.png'), dpi=130, bbox_inches='tight')
plt.close(fig)
print('charts written to', ASSETS)
