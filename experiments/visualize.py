"""
Run one experiment, record every spike, and render a dashboard PNG:
  1. spike raster of active neurons sorted by first-spike latency (cascade view)
  2. population firing rate over time
  3. per-neuron mean firing-rate distribution
  4. propagation: number of newly recruited neurons per 10 ms bin

Usage:
    python experiments/visualize.py --exp sugar --t 1.0
    python experiments/visualize.py --exp p9 --t 1.0
Output: experiments/out/<exp>_t<t>s.png (+ spikes parquet)
"""
import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

import pyarrow  # noqa: F401
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmark import get_experiment, path_comp, path_con, path_wt
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights


def simulate(exp_key, t_sec, device):
    exp = get_experiment(exp_key)
    flyid2i, i2flyid = get_hash_tables(str(path_comp))
    exc_idx = [flyid2i[n] for n in exp['neu_exc']]
    weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(device)
    n = weights.shape[0]
    model = TorchModel(1, n, DT, MODEL_PARAMS, weights, exc_indices=exc_idx, device=device)
    state = model.state_init()
    rates = torch.zeros(1, n, device=device)
    rates[:, exc_idx] = exp['stim_rate']

    steps = int(t_sec * 1000 / DT)
    t_list, n_list = [], []
    t0 = perf_counter()
    with torch.no_grad():
        for step in range(steps):
            state = model(rates, *state)
            idx = state[2][0].nonzero(as_tuple=True)[0]
            if idx.numel():
                n_list.append(idx.cpu())
                t_list.append(torch.full((idx.numel(),), step * DT))
    print(f'simulated {t_sec}s in {perf_counter()-t0:.1f}s')
    spikes = pd.DataFrame({
        'time_ms': torch.cat(t_list).numpy(),
        'neuron': torch.cat(n_list).numpy(),
    })
    spikes['stimulated'] = spikes['neuron'].isin(exc_idx)
    spikes['flywire_id'] = spikes['neuron'].map(i2flyid)
    return exp, spikes, n


def plot(exp, spikes, n_total, t_sec, out_png):
    t_ms = t_sec * 1000
    first = spikes.groupby('neuron')['time_ms'].min().sort_values()
    order = {nid: r for r, nid in enumerate(first.index)}
    spikes['row'] = spikes['neuron'].map(order)
    n_active = len(first)
    rate = spikes.groupby('neuron').size() / t_sec

    fig = plt.figure(figsize=(14, 10), dpi=130)
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.2], hspace=0.45, wspace=0.25)

    # 1. raster
    ax = fig.add_subplot(gs[0, :])
    stim = spikes['stimulated']
    ax.scatter(spikes.loc[~stim, 'time_ms'], spikes.loc[~stim, 'row'],
               s=1.5, c='#1f77b4', marker='|', linewidths=0.6, label='downstream')
    ax.scatter(spikes.loc[stim, 'time_ms'], spikes.loc[stim, 'row'],
               s=1.5, c='#d62728', marker='|', linewidths=0.6, label='stimulated')
    ax.set_xlim(0, t_ms)
    ax.set_ylim(-1, n_active)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('neuron (sorted by first-spike latency)')
    ax.set_title(f'{exp["name"]}  |  {n_active} of {n_total:,} neurons active  |  '
                 f'{len(spikes):,} spikes in {t_sec:.1f} s')
    ax.legend(loc='lower right', markerscale=6)

    # 2. population rate
    ax = fig.add_subplot(gs[1, :])
    bins = np.arange(0, t_ms + 5, 5)
    counts, _ = np.histogram(spikes.loc[~stim, 'time_ms'], bins=bins)
    ax.plot(bins[:-1], counts / 0.005, color='#1f77b4', lw=1)
    ax.set_xlim(0, t_ms)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('downstream spikes / s')
    ax.set_title('population activity (downstream only, 5 ms bins)')

    # 3. rate distribution
    ax = fig.add_subplot(gs[2, 0])
    ds_rate = rate[~rate.index.isin(spikes.loc[stim, 'neuron'].unique())]
    ax.hist(ds_rate, bins=40, color='#1f77b4')
    ax.set_xlabel('mean firing rate (Hz)')
    ax.set_ylabel('# neurons')
    ax.set_title(f'downstream rate distribution (median {ds_rate.median():.1f} Hz)')

    # 4. recruitment
    ax = fig.add_subplot(gs[2, 1])
    ds_first = first[~first.index.isin(spikes.loc[stim, 'neuron'].unique())]
    bins = np.arange(0, min(t_ms, 300) + 10, 10)
    ax.hist(ds_first, bins=bins, color='#2ca02c')
    ax.set_xlabel('first-spike latency (ms)')
    ax.set_ylabel('# neurons recruited')
    ax.set_title('recruitment cascade (first 300 ms)')

    fig.savefig(out_png, bbox_inches='tight')
    print('wrote', out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='sugar', choices=['sugar', 'p9'])
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = ROOT / 'experiments' / 'out'
    out_dir.mkdir(exist_ok=True)
    exp, spikes, n = simulate(args.exp, args.t, args.device)
    spikes.to_parquet(out_dir / f'{args.exp}_t{args.t}s_spikes.parquet')
    plot(exp, spikes, n, args.t, out_dir / f'{args.exp}_t{args.t}s.png')


if __name__ == '__main__':
    main()
