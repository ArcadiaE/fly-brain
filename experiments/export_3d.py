"""
Run an experiment and export a self-contained 3D replay page.

Records every spike plus membrane-voltage traces of the stimulated neurons and
the most active downstream neurons, joins FlyWire soma/anchor coordinates and
cell-type annotations, and injects everything into brain3d_template.html.

Usage:
    python experiments/export_3d.py --exp sugar --t 1.0
Output:
    experiments/out/brain3d_<exp>.html   (open in any browser)
"""
import argparse
import base64
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

import pyarrow  # noqa: F401
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from benchmark import get_experiment, path_comp, path_con, path_wt
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights

ANN_PATH = ROOT / 'data/annotations/Supplemental_file1_neuron_annotations.tsv'
TEMPLATE = ROOT / 'experiments/brain3d_template.html'
N_TRACE_DOWNSTREAM = 8
N_TRACE_STIM = 4


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode('ascii')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='sugar', choices=['sugar', 'p9'])
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--silence', default='',
                    help='cell_type prefix to silence, e.g. MBON; matching neurons never spike')
    args = ap.parse_args()
    dev = args.device

    exp = get_experiment(args.exp)
    flyid2i, i2flyid = get_hash_tables(str(path_comp))
    exc_idx = [flyid2i[n] for n in exp['neu_exc']]
    weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(dev)
    n = weights.shape[0]

    ann = pd.read_csv(ANN_PATH, sep='\t', low_memory=False).set_index('root_id')
    ids = np.array([i2flyid[i] for i in range(n)], dtype=np.int64)
    a = ann.reindex(ids)

    sil_idx = []
    alive = None
    if args.silence:
        ct = a['cell_type'].fillna(a['hemibrain_type']).fillna('').astype(str)
        sil_idx = np.nonzero(ct.str.startswith(args.silence).to_numpy())[0].tolist()
        if not sil_idx:
            raise SystemExit(f'no neuron matches --silence {args.silence!r}')
        print(f'silencing {len(sil_idx)} neurons with type prefix {args.silence!r}')
        alive = torch.ones(1, n, device=dev)
        alive[0, sil_idx] = 0
    model = TorchModel(1, n, DT, MODEL_PARAMS, weights, exc_indices=exc_idx, device=dev)
    state = model.state_init()
    rates = torch.zeros(1, n, device=dev)
    rates[:, exc_idx] = exp['stim_rate']
    steps = int(args.t * 1000 / DT)

    # ---- pass 1: simulate, record spikes and full voltage only for a probe set ----
    # We do not know the top downstream neurons in advance, so record voltage for
    # everyone into a rolling GPU buffer is too big (138k x 10k floats = 5.5 GB).
    # Instead: run once to find the top downstream neurons, then run again with a
    # fixed seed recording only those. Both runs use the same generator seed.
    def run(record_idx=None, seed=0):
        gen = torch.Generator(device=dev).manual_seed(seed)
        st = model.state_init()
        t_list, n_list = [], []
        traces = None
        if record_idx is not None:
            rec = torch.as_tensor(record_idx, device=dev)
            traces = torch.zeros(len(record_idx), steps, device=dev)
        with torch.no_grad():
            for step in tqdm(range(steps), ncols=90, unit='step'):
                st = model(rates, *st, generator=gen)
                if alive is not None:
                    st = (st[0], st[1], st[2] * alive, st[3], st[4])
                idx = st[2][0].nonzero(as_tuple=True)[0]
                if idx.numel():
                    n_list.append(idx.cpu())
                    t_list.append(torch.full((idx.numel(),), step, dtype=torch.int32))
                if traces is not None:
                    traces[:, step] = st[3][0, rec]
        sp_t = torch.cat(t_list).numpy() if t_list else np.zeros(0, np.int32)
        sp_n = torch.cat(n_list).numpy() if n_list else np.zeros(0, np.int64)
        return sp_t, sp_n, (traces.cpu().numpy() if traces is not None else None)

    print('pass 1: find most active downstream neurons')
    t0 = perf_counter()
    sp_t, sp_n, _ = run()
    counts = np.bincount(sp_n, minlength=n)
    counts_ds = counts.copy()
    counts_ds[exc_idx] = 0
    top_ds = np.argsort(-counts_ds)[:N_TRACE_DOWNSTREAM]
    trace_idx = list(exc_idx[:N_TRACE_STIM]) + [int(i) for i in top_ds]

    print('pass 2: record voltage traces')
    sp_t, sp_n, traces = run(record_idx=trace_idx)
    print(f'simulated twice in {perf_counter()-t0:.1f}s; {len(sp_t)} spikes, '
          f'{(np.bincount(sp_n, minlength=n) > 0).sum()} active neurons')

    # ---- annotations & coordinates ----
    xyz = a[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=np.float32)
    xyz *= np.array([4, 4, 40], np.float32) / 1000.0          # voxels -> microns
    missing = np.isnan(xyz).any(axis=1)
    xyz[missing] = np.nan
    center = np.nanmean(xyz, axis=0)
    xyz -= center
    xyz[missing] = 1e6  # park unknowns far away
    classes = ['optic', 'central', 'sensory', 'visual_projection', 'ascending',
               'descending', 'sensory_ascending', 'visual_centrifugal', 'motor',
               'endocrine', 'unknown']
    sc = a['super_class'].fillna('unknown').map({c: i for i, c in enumerate(classes)})
    sc = sc.fillna(len(classes) - 1).to_numpy(dtype=np.uint8)

    active = np.unique(sp_n)
    def info(i):
        r = a.iloc[i]
        return {
            'id': str(ids[i]),
            'type': str(r.get('cell_type') or r.get('hemibrain_type') or ''),
            'cls': str(r.get('cell_class') or ''),
            'sc': str(r.get('super_class') or ''),
            'nt': str(r.get('top_nt') or ''),
            'side': str(r.get('side') or ''),
            'rate': float(counts[i] / args.t),
        }
    active_info = {int(i): info(int(i)) for i in active}
    trace_info = [dict(info(i), idx=int(i), stim=(i in set(exc_idx))) for i in trace_idx]

    order = np.argsort(sp_t, kind='stable')
    tag = f' + {args.silence} silenced' if args.silence else ''
    payload = {
        'exp': exp['name'] + tag,
        'n': int(n),
        'dt': DT,
        'steps': steps,
        'stim_rate': exp['stim_rate'],
        'stim_idx': [int(i) for i in exc_idx],
        'sil_idx': sil_idx,
        'classes': classes,
        'pos_b64': b64(xyz.astype(np.float32)),
        'sc_b64': b64(sc),
        'spike_t_b64': b64(sp_t[order].astype(np.uint16)),
        'spike_n_b64': b64(sp_n[order].astype(np.uint32)),
        'trace_b64': b64(traces.astype(np.float32)),
        'traces': trace_info,
        'v_th': MODEL_PARAMS['vThreshold'],
        'v_rest': MODEL_PARAMS['vRest'],
        'active': active_info,
    }
    html = TEMPLATE.read_text(encoding='utf-8').replace(
        '/*__DATA__*/null', json.dumps(payload))
    out = ROOT / 'experiments/out' / f'brain3d_{args.exp}{"_no" + args.silence.lower() if args.silence else ""}.html'
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print(f'wrote {out}  ({out.stat().st_size/2**20:.1f} MB)')


if __name__ == '__main__':
    main()
