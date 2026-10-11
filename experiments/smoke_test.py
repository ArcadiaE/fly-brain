"""
Smoke test: load the FlyWire connectome into the PyTorch LIF model and run a
short simulation on whichever device is available.

Usage (from repo root, inside .venv):
    python experiments/smoke_test.py                # sugar GRNs, 1 s, cuda if available
    python experiments/smoke_test.py --exp p9 --t 0.5
    python experiments/smoke_test.py --device cpu
"""
import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

import pyarrow  # noqa: F401  (import before torch, see run_pytorch.py)
import pandas as pd
import torch

from benchmark import get_experiment, path_comp, path_con, path_wt
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='sugar', choices=['sugar', 'p9'])
    ap.add_argument('--t', type=float, default=1.0, help='simulated seconds')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = args.device
    print(f'torch {torch.__version__}  device={dev}', end='')
    if dev == 'cuda':
        print(f'  ({torch.cuda.get_device_name(0)})')
    else:
        print()

    exp = get_experiment(args.exp)
    print(f'experiment: {exp["name"]}  stim_rate={exp["stim_rate"]} Hz')

    t0 = perf_counter()
    flyid2i, i2flyid = get_hash_tables(str(path_comp))
    exc_idx = [flyid2i[n] for n in exp['neu_exc']]
    weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(dev)
    n = weights.shape[0]
    print(f'weights: {n} neurons, {weights._nnz()} synapses, load {perf_counter()-t0:.1f}s')

    model = TorchModel(args.batch, n, DT, MODEL_PARAMS, weights, exc_indices=exc_idx, device=dev)
    state = model.state_init()
    rates = torch.zeros(args.batch, n, device=dev)
    rates[:, exc_idx] = exp['stim_rate']

    steps = int(args.t * 1000 / DT)
    counts = torch.zeros(n, device=dev)
    if dev == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    from tqdm import tqdm
    print(f'running {steps} steps of {DT} ms  (= {args.t:.2f} s of brain time)')
    t0 = perf_counter()
    bar = tqdm(range(steps), unit='step', ncols=100)
    with torch.no_grad():
        for step in bar:
            state = model(rates, *state)
            counts += state[2][0]
            if step % 200 == 0:
                bar.set_postfix(brain_ms=f'{step*DT:.0f}',
                                spikes=int(counts.sum()),
                                active=int((counts > 0).sum()))
    if dev == 'cuda':
        torch.cuda.synchronize()
    sim = perf_counter() - t0

    print(f'simulated {args.t:.2f}s in {sim:.2f}s wall  '
          f'({args.t/sim:.3f}x realtime, {sim/steps*1000:.2f} ms/step)')
    if dev == 'cuda':
        print(f'peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GB')

    counts = counts.cpu()
    active = (counts > 0).sum().item()
    print(f'active neurons: {active}   total spikes: {int(counts.sum())}')
    exc_set = set(exc_idx)
    downstream = counts.clone()
    downstream[exc_idx] = 0
    top = torch.topk(downstream, 15)
    df = pd.DataFrame({
        'flywire_id': [i2flyid[int(i)] for i in top.indices],
        'rate_hz': (top.values / args.t).numpy().round(1),
    })
    print(f'stimulated neurons mean rate: {counts[exc_idx].mean()/args.t:.1f} Hz')
    print('top downstream (non-stimulated) neurons:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
