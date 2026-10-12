"""Generate publication-quality art images from real project data.

Outputs (docs/assets/):
  hero_brain.png      - all 138k neuron somas colored by super_class + fired-neuron glow
  raster_sugar.png    - spike raster of the sugar experiment, colored by first-spike latency
  connectome_web.png  - sampled synapse edges drawn as a luminous web
  pipeline_en.png / pipeline_zh.png - schematic pipeline diagram (EN / ZH labels)

Data mapping note: neuron index i in the connectivity/simulation matrices equals
row i of data/2025_Completeness_783.csv (root_ids sorted ascending). Coordinates
come from the annotations TSV (soma_x/y/z, falling back to pos_x/y/z when the soma
is unannotated), converted from 4x4x40 nm voxels to microns.

Run: python experiments/make_art.py [hero|raster|web|pipeline|all]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
ANN_PATH = ROOT / 'data/annotations/Supplemental_file1_neuron_annotations.tsv'
COMP_PATH = ROOT / 'data/2025_Completeness_783.csv'
CONN_PATH = ROOT / 'data/2025_Connectivity_783.parquet'
SPIKE_PATH = ROOT / 'data/results/local_fastbrain/round_01/fastbrain_dt0.25_t0.5s_n1.parquet'
OUT_DIR = ROOT / 'docs/assets'

BG = '#0a0e1a'

CLASS_COLORS = {
    'optic':              '#5b8ff9',
    'visual_projection':  '#64c2f5',
    'visual_centrifugal': '#4f7dd0',
    'central':            '#9d6ff5',
    'sensory':            '#43d9b8',
    'motor':              '#f6a35c',
    'descending':         '#e8684a',
    'ascending':          '#f6c85c',
    'sensory_ascending':  '#f6c85c',
    'endocrine':          '#c9b8d9',
    'unknown':            '#5a6478',
}

# Frontal view: x = left-right, z = dorsal-ventral (y is anterior-posterior depth).
# Voxel coords have z increasing ventrally, so flip z to put dorsal up.
# A slight tilt around the left-right axis adds depth and spreads out the sheet of
# ascending neurons (their anchors genuinely lie on a thin horizontal plane at the
# neck connective, which would otherwise render as a straight line).
VIEW_AXES = (0, 2)          # (horizontal, vertical) columns of xyz
FLIP = (True, True)         # flip x so left hemisphere is on the left; flip z so dorsal is up
TILT_RAD = np.radians(12)   # rotate around the x (left-right) axis


def load_geometry():
    """Return (xyz_um [n,3] with NaN where unknown, super_class [n] of str)."""
    comp = pd.read_csv(COMP_PATH, index_col=0)
    ids = comp.index.to_numpy()
    ann = pd.read_csv(ANN_PATH, sep='\t', low_memory=False,
                      usecols=['root_id', 'soma_x', 'soma_y', 'soma_z',
                               'pos_x', 'pos_y', 'pos_z', 'super_class']).set_index('root_id')
    a = ann.reindex(ids)
    soma = a[['soma_x', 'soma_y', 'soma_z']].to_numpy(float)
    pos = a[['pos_x', 'pos_y', 'pos_z']].to_numpy(float)
    use_soma = ~np.isnan(soma).any(axis=1)
    xyz = np.where(use_soma[:, None], soma, pos)
    xyz = xyz * np.array([4.0, 4.0, 40.0]) / 1000.0   # voxels -> microns
    xyz -= np.nanmean(xyz, axis=0)
    sc = a['super_class'].fillna('unknown').to_numpy()
    return xyz, sc


def project(xyz):
    """Project 3D coords to the tilted frontal 2D view, returning (x2d, y2d)."""
    y = xyz[:, 1]
    z = xyz[:, 2]
    z_rot = y * np.sin(TILT_RAD) + z * np.cos(TILT_RAD)
    x2d = xyz[:, VIEW_AXES[0]].copy()
    y2d = z_rot
    if FLIP[0]:
        x2d = -x2d
    if FLIP[1]:
        y2d = -y2d
    return x2d, y2d


def blank_fig(w_in, h_in, bg=BG):
    fig = plt.figure(figsize=(w_in, h_in), dpi=150, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    ax.axis('off')
    return fig, ax


def load_fired():
    """Return neuron indices that fired in the sugar experiment, and the spike table."""
    sp = pd.read_parquet(SPIKE_PATH)
    fired = np.sort(sp['neuron_index'].unique())
    return fired, sp


# ---------------------------------------------------------------- hero

def make_hero():
    xyz, sc = load_geometry()
    ok = ~np.isnan(xyz).any(axis=1)
    x2d, y2d = project(xyz)
    fired, _ = load_fired()

    fig, ax = blank_fig(12.8, 7.2)

    # draw dim classes first, bright ones last
    draw_order = ['unknown', 'endocrine', 'sensory_ascending', 'ascending',
                  'descending', 'motor', 'sensory', 'visual_centrifugal',
                  'visual_projection', 'central', 'optic']
    for cls in draw_order:
        m = ok & (sc == cls)
        if not m.any():
            continue
        c = CLASS_COLORS[cls]
        # halo pass + core pass for a soft glow
        ax.scatter(x2d[m], y2d[m], s=2.2, c=c, alpha=0.10, linewidths=0, rasterized=True)
        ax.scatter(x2d[m], y2d[m], s=0.55, c=c, alpha=0.75, linewidths=0, rasterized=True)

    # fired neurons: big low-alpha halo + bright warm core
    f = fired[ok[fired]]
    ax.scatter(x2d[f], y2d[f], s=24, c='#ffd76a', alpha=0.14, linewidths=0, rasterized=True)
    ax.scatter(x2d[f], y2d[f], s=7, c='#ffe9a8', alpha=0.85, linewidths=0, rasterized=True)
    ax.scatter(x2d[f], y2d[f], s=2.2, c='#fffdf4', alpha=1.0, linewidths=0, rasterized=True)

    # brain fills most of the frame
    cx, cy = x2d[ok].mean(), y2d[ok].mean()
    span = (x2d[ok].max() - x2d[ok].min()) / 0.86
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2 * (7.2 / 12.8), cy + span / 2 * (7.2 / 12.8))

    out = OUT_DIR / 'hero_brain.png'
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print('wrote', out)


# ---------------------------------------------------------------- raster

def make_raster():
    _, sp = load_fired()
    first = sp.groupby('neuron_index')['time_ms'].min()
    order = first.sort_values().index.to_numpy()
    rank = pd.Series(np.arange(len(order)), index=order)
    y = sp['neuron_index'].map(rank).to_numpy()
    lat = sp['neuron_index'].map(first).to_numpy()

    fig = plt.figure(figsize=(10.67, 6.0), dpi=150, facecolor=BG)
    ax = fig.add_axes([0.07, 0.11, 0.90, 0.85])
    ax.set_facecolor(BG)
    ax.scatter(sp['time_ms'], y, s=7, c=lat, cmap='turbo', alpha=0.9,
               linewidths=0, rasterized=True, vmin=0, vmax=350)
    ax.set_xlim(0, sp['time_ms'].max() * 1.02)
    ax.set_ylim(-1, len(order))
    ax.set_xlabel('time (ms)', color='#aab3c5', fontsize=11)
    ax.set_ylabel('neuron (sorted by first spike)', color='#aab3c5', fontsize=11)
    ax.tick_params(colors='#8a94a8', labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#3a4358')
    ax.grid(False)

    out = OUT_DIR / 'raster_sugar.png'
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print('wrote', out)


# ---------------------------------------------------------------- connectome web

def make_web(n_edges=80_000, seed=7, min_len_um=4.0):
    xyz, _ = load_geometry()
    ok = ~np.isnan(xyz).any(axis=1)
    x2d, y2d = project(xyz)

    tbl = pq.read_table(CONN_PATH, columns=['Presynaptic_Index', 'Postsynaptic_Index'])
    pre = tbl.column('Presynaptic_Index').to_numpy()
    post = tbl.column('Postsynaptic_Index').to_numpy()
    keep = ok[pre] & ok[post]
    pre, post = pre[keep], post[keep]
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pre), size=min(n_edges, len(pre)), replace=False)
    pre, post = pre[sel], post[sel]
    # drop degenerate ultra-short edges (invisible at this scale; dense clusters of
    # them, e.g. inside the ascending-neuron sheet, render as scratches of light)
    seg_len = np.hypot(x2d[pre] - x2d[post], y2d[pre] - y2d[post])
    keep2 = seg_len >= min_len_um
    pre, post = pre[keep2], post[keep2]

    from matplotlib.collections import LineCollection
    segs = np.stack([
        np.column_stack([x2d[pre], y2d[pre]]),
        np.column_stack([x2d[post], y2d[post]]),
    ], axis=1)
    lc = LineCollection(segs, colors=(0.42, 0.78, 1.0, 0.05), linewidths=0.3,
                        rasterized=True, capstyle='round')

    fig, ax = blank_fig(12.8, 7.2)
    ax.add_collection(lc)
    ax.scatter(x2d[ok], y2d[ok], s=0.1, c='#8ec9f0', alpha=0.10,
               linewidths=0, rasterized=True)

    cx, cy = x2d[ok].mean(), y2d[ok].mean()
    span = (x2d[ok].max() - x2d[ok].min()) / 0.80
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2 * (7.2 / 12.8), cy + span / 2 * (7.2 / 12.8))

    out = OUT_DIR / 'connectome_web.png'
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print('wrote', out)


# ---------------------------------------------------------------- pipeline diagram

PIPELINE = {
    'en': [
        ('Sensory stimulus', '(sugar / P9 leg taste)'),
        ('Whole-brain LIF simulation', '138,639 neurons - FlyWire v783'),
        ('FastBrain engine', 'event-driven - CUDA Graphs'),
        ('Spike readout & analysis', 'rasters - benchmarks'),
    ],
    'zh': [
        ('感觉刺激', '（蔗糖 / P9 腿味觉）'),
        ('全脑 LIF 仿真', '138,639 个神经元 · FlyWire v783'),
        ('FastBrain 引擎', '事件驱动 · CUDA Graph'),
        ('脉冲读出与分析', '脉冲光栅 · 基准测试'),
    ],
}


def make_pipeline(lang):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    if lang == 'zh':
        plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

    fig = plt.figure(figsize=(10.67, 3.33), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)

    accent = '#4C72B0'
    n = len(PIPELINE[lang])
    box_w, box_h, y0 = 3.2, 1.9, 1.55
    gap = (16 - n * box_w) / (n + 1)

    for i, (title, sub) in enumerate(PIPELINE[lang]):
        x0 = gap + i * (box_w + gap)
        box = FancyBboxPatch((x0, y0), box_w, box_h,
                             boxstyle='round,pad=0.12,rounding_size=0.22',
                             facecolor='#131b2e', edgecolor=accent, linewidth=2.0)
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, y0 + box_h * 0.62, title,
                ha='center', va='center', color='#e8edf7', fontsize=12.5,
                fontweight='bold')
        ax.text(x0 + box_w / 2, y0 + box_h * 0.28, sub,
                ha='center', va='center', color='#9aa7c0', fontsize=9)
        if i < n - 1:
            x_start = x0 + box_w + 0.16
            x_end = x0 + box_w + gap - 0.16
            ax.add_patch(FancyArrowPatch((x_start, y0 + box_h / 2),
                                         (x_end, y0 + box_h / 2),
                                         arrowstyle='-|>', mutation_scale=22,
                                         color=accent, linewidth=2.2))

    out = OUT_DIR / f'pipeline_{lang}.png'
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    plt.rcParams['font.family'] = 'sans-serif'
    print('wrote', out)


# ---------------------------------------------------------------- main

if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('hero', 'all'):
        make_hero()
    if which in ('raster', 'all'):
        make_raster()
    if which in ('web', 'all'):
        make_web()
    if which in ('pipeline', 'all'):
        make_pipeline('en')
        make_pipeline('zh')
