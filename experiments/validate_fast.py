"""
Compare FastBrain against the reference TorchModel (code/run_pytorch.py).

Runs the reference twice with different seeds (to measure run-to-run
variability), then FastBrain at the given dt values, and reports per-neuron
firing-rate agreement.

Usage:
    python experiments/validate_fast.py --t 1.0 --dt 0.1 0.25 0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'experiments'))
import pyarrow  # noqa: F401,E402
from benchmark import get_experiment, path_comp, path_con, path_wt  # noqa: E402
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights  # noqa: E402
from fast_brain import FastBrain  # noqa: E402


def run_reference(exc, rate, t_sec, seed):
    dev = 'cuda'
    torch.manual_seed(seed)
    w = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(dev)
    n = w.shape[0]
    model = TorchModel(1, n, DT, MODEL_PARAMS, w, exc_indices=exc, device=dev)
    st = model.state_init()
    rates = torch.zeros(1, n, device=dev); rates[:, exc] = rate
    counts = torch.zeros(n, device=dev)
    with torch.no_grad():
        for _ in range(int(t_sec * 1000 / DT)):
            st = model(rates, *st)
            counts += st[2][0]
    del model, w
    torch.cuda.empty_cache()
    return counts.cpu().numpy()


def run_fast(exc, rate, t_sec, dt, seed):
    fb = FastBrain(dt=dt, seed=seed)
    fb.set_rate(exc, rate)
    counts = np.zeros(fb.n, np.int64)
    blocks = int(round(t_sec * 1000 / dt / fb.S))
    for _ in range(blocks):
        fb.run_block()
        sp, _ = fb.block_spikes_host()
        for ids in sp:
            counts[ids] += 1
    del fb
    torch.cuda.empty_cache()
    return counts


def compare(a, b, t_sec, exc, label):
    ra, rb = a / t_sec, b / t_sec
    act_a, act_b = ra > 0, rb > 0
    union = act_a | act_b
    jac = (act_a & act_b).sum() / max(1, union.sum())
    corr = np.corrcoef(ra[union], rb[union])[0, 1] if union.sum() > 2 else float('nan')
    mask = np.ones_like(ra, bool); mask[exc] = False
    print(f'{label:28s} active {act_a.sum():4d} vs {act_b.sum():4d}  jaccard {jac:.3f}  '
          f'rate corr {corr:.3f}  downstream spikes {a[mask].sum():.0f} vs {b[mask].sum():.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--dt', type=float, nargs='+', default=[0.1, 0.25, 0.5])
    ap.add_argument('--exp', default='sugar')
    args = ap.parse_args()
    exp = get_experiment(args.exp)
    f2i, _ = get_hash_tables(str(path_comp))
    exc = [f2i[i] for i in exp['neu_exc']]
    rate = exp['stim_rate']

    print('reference run A/B ...')
    refA = run_reference(exc, rate, args.t, seed=1)
    refB = run_reference(exc, rate, args.t, seed=2)
    compare(refA, refB, args.t, exc, 'reference A vs B')
    for dt in args.dt:
        fast = run_fast(exc, rate, args.t, dt, seed=3)
        compare(refA, fast, args.t, exc, f'reference vs fast dt={dt}')


if __name__ == '__main__':
    main()
