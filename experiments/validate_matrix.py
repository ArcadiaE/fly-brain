"""
Validation matrix: FastBrain vs the reference TorchModel (code/run_pytorch.py).

Extends validate_fast.py from a single condition to a matrix over
experiments x durations x seeds x dt values, and adds spike-time metrics.

Metrics per comparison (a vs b, "active" = spike count > 0, same definition
as validate_fast.py):
  * rate_corr        Pearson corr of per-neuron firing rates on the union of
                     active neurons
  * jaccard          |active_a & active_b| / |active_a | active_b|
  * count_ratio      total spikes a / total spikes b
  * spike_precision  fraction of a's spikes that have a b spike from the SAME
                     neuron within +-PRECISION_WINDOW_MS ms. Existence-based:
                     each a spike independently checks the sorted spike times
                     of b (np.searchsorted); a single b spike may cover
                     several a spikes. Simple and symmetric enough for a
                     +-2 ms window.

Comparison kinds (CSV column "kind"):
  * ref-floor        ref(seed[0]) vs ref(other seed): run-to-run stochasticity
                     of the reference itself -- the noise floor every FastBrain
                     number should be judged against
  * fast-matched     fast(seed=s) vs ref(seed=s). NOTE: same numeric seed does
                     NOT reproduce the same Poisson draws (the two engines
                     consume the default RNG stream differently), so this is
                     only marginally more "matched" than fast-fixed.
  * fast-fixed       fast(seed=s) vs ref(seeds[0])
  * drift            per-1 s-window rate corr of the long-run drift check

Seed handling: the reference is reseeded via torch.manual_seed(seed) per run.
FastBrain takes the seed in its constructor; because a captured CUDA graph
owns its RNG state (re-seeding after capture has no effect on replays),
a NEW FastBrain instance is built per (dt, seed) -- graph capture included.

Spike-time convention: a spike generated during step i is stamped (i+1)*dt ms
(reference) / j*dt ms where j is the global index in the concatenated
block_spikes_host() list (FastBrain, which reports spikes of step j-1 at
list index j). Both mean (generation_step+1)*dt, so they are aligned.

FastBrain simulates ceil(t / (S*dt)) whole blocks; rates are computed with the
ACTUAL simulated time t_fast = blocks*S*dt/1000 s.

GPU contention: per NOTES.md, live_server.py must not run during benchmarks
(it roughly doubles kernel timings). The script scans the process list and
aborts if a live_server.py python process is found (override with --force).

Usage:
    python experiments/validate_matrix.py --quick     # tiny smoke config
    python experiments/validate_matrix.py --full      # whole matrix (long)
    python experiments/validate_matrix.py --full --seeds 1 2 3 --durations 0.5
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'experiments'))
import pyarrow  # noqa: F401,E402
from benchmark import get_experiment, path_comp, path_con, path_wt  # noqa: E402
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights  # noqa: E402
from fast_brain import FastBrain  # noqa: E402

PRECISION_WINDOW_MS = 2.0
REC_CHUNK = 500          # reference spike-recording chunk (steps); 500 x 138639 bool ~ 69 MB
OUT_DIR = ROOT / 'experiments' / 'out' / 'validation'

QUICK = dict(exps=['sugar'], durations=[0.2], seeds=[1], dts=[0.1, 0.25], drift=False)
FULL = dict(exps=['sugar', 'p9'], durations=[0.5, 1.0], seeds=[1, 2, 3, 4, 5],
            dts=[0.1, 0.25, 0.5], drift=True)
DRIFT_CFG = dict(exp='sugar', t_sec=10.0, dt=0.25, window_sec=1.0, seed=1)


# ---------------------------------------------------------------- GPU guard
def find_live_server():
    """Return [(pid, cmdline)] of python processes running live_server.py."""
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
             "| Select-Object ProcessId,CommandLine | Format-List"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        print(f'warning: process scan failed ({e}); cannot rule out live_server')
        return []
    hits = []
    for block in out.split('\n\n'):
        if 'live_server.py' in block:
            m = re.search(r'ProcessId\s*:\s*(\d+)', block)
            cmd = re.search(r'CommandLine\s*:\s*(.*)', block)
            hits.append((m.group(1) if m else '?',
                         cmd.group(1).strip() if cmd else ''))
    return hits


# ---------------------------------------------------------------- runners
def run_reference(weights, n, exc, rate, t_sec, seed,
                  record_times=True, window_sec=None):
    """Reference TorchModel at DT=0.1 ms.

    Returns dict(counts=int64[n], times={nid: ms array} or None,
                 windows=int64[n_win, n] or None).
    Spikes are recorded in REC_CHUNK-step bool buffers on GPU and fetched once
    per chunk, avoiding a host sync every step.
    """
    dev = 'cuda'
    torch.manual_seed(seed)
    model = TorchModel(1, n, DT, MODEL_PARAMS, weights, exc_indices=exc, device=dev)
    st = model.state_init()
    rates = torch.zeros(1, n, device=dev)
    rates[:, exc] = rate
    steps = int(round(t_sec * 1000 / DT))
    counts = torch.zeros(n, dtype=torch.long, device=dev)
    buf = torch.zeros(REC_CHUNK, n, dtype=torch.bool, device=dev)
    tt, nn = [], []
    win_steps = int(round(window_sec * 1000 / DT)) if window_sec else 0
    win_counts = torch.zeros(n, dtype=torch.long, device=dev) if win_steps else None
    windows = []
    with torch.no_grad():
        for step in range(steps):
            st = model(rates, *st)
            spk = st[2][0] > 0
            counts += spk
            if record_times:
                c = step % REC_CHUNK
                buf[c] = spk
                if c == REC_CHUNK - 1 or step == steps - 1:
                    idx = buf[:c + 1].nonzero()
                    tt.append((idx[:, 0].cpu().numpy() + (step - c) + 1) * DT)
                    nn.append(idx[:, 1].cpu().numpy())
            if win_steps:
                win_counts += spk
                if (step + 1) % win_steps == 0:
                    windows.append(win_counts.cpu().numpy().copy())
                    win_counts.zero_()
    del model, st, buf
    torch.cuda.empty_cache()
    times = _times_dict(np.concatenate(tt), np.concatenate(nn)) if record_times else None
    return dict(counts=counts.cpu().numpy(), times=times,
                windows=np.stack(windows) if windows else None,
                t_actual=steps * DT / 1000.0)


def run_fast(exc, rate, t_sec, dt, seed, record_times=True, window_sec=None):
    """FastBrain at the given dt; one fresh instance (graph capture) per call.

    Returns the same dict as run_reference, plus overflow_chunks.
    """
    fb = FastBrain(dt=dt, seed=seed)
    fb.set_rate(exc, rate)
    blocks = int(round(t_sec * 1000 / dt / fb.S))
    counts = np.zeros(fb.n, np.int64)
    tt, nn = [], []
    win_blocks = int(round(window_sec * 1000 / dt / fb.S)) if window_sec else 0
    win_counts = np.zeros(fb.n, np.int64) if win_blocks else None
    windows = []
    overflow = 0
    for b in range(blocks):
        fb.run_block()
        sp, of = fb.block_spikes_host()
        overflow += of
        for si, ids in enumerate(sp):
            counts[ids] += 1
            if record_times and len(ids):
                tt.append(np.full(len(ids), (b * fb.S + si) * dt))
                nn.append(ids)
            if win_blocks:
                win_counts[ids] += 1
        if win_blocks and (b + 1) % win_blocks == 0:
            windows.append(win_counts.copy())
            win_counts[:] = 0
    n, t_actual = fb.n, blocks * fb.S * dt / 1000.0
    del fb
    torch.cuda.empty_cache()
    times = _times_dict(np.concatenate(tt), np.concatenate(nn)) if record_times and tt else None
    return dict(counts=counts, times=times,
                windows=np.stack(windows) if windows else None,
                t_actual=t_actual, overflow=overflow)


def _times_dict(times_ms, neuron_ids):
    """{neuron_id: sorted spike-time array (ms)} for active neurons only."""
    d = {}
    for t, nid in zip(times_ms, neuron_ids):
        d.setdefault(int(nid), []).append(float(t))
    return {k: np.asarray(v) for k, v in d.items()}


# ---------------------------------------------------------------- metrics
def spike_precision(times_a, times_b, window=PRECISION_WINDOW_MS):
    """Fraction of a's spikes with a same-neuron b spike within +-window ms."""
    total, hit = 0, 0
    for nid, ta in times_a.items():
        total += len(ta)
        tb = times_b.get(nid)
        if tb is None:
            continue
        lo = np.searchsorted(tb, ta - window, 'left')
        hi = np.searchsorted(tb, ta + window, 'right')
        hit += int((hi > lo).sum())
    return hit / total if total else float('nan')


def compare(a, b, exc, label):
    """a, b: result dicts from run_reference/run_fast. Returns a metrics row."""
    ca, cb = a['counts'], b['counts']
    ra, rb = ca / a['t_actual'], cb / b['t_actual']
    act_a, act_b = ra > 0, rb > 0
    union = act_a | act_b
    jac = float((act_a & act_b).sum() / max(1, union.sum()))
    corr = float(np.corrcoef(ra[union], rb[union])[0, 1]) if union.sum() > 2 else float('nan')
    ratio = float(ca.sum() / max(1, cb.sum()))
    prec = (spike_precision(a['times'], b['times'])
            if a['times'] is not None and b['times'] is not None else float('nan'))
    row = dict(rate_corr=corr, jaccard=jac, count_ratio=ratio, spike_precision=prec,
               active_a=int(act_a.sum()), active_b=int(act_b.sum()),
               spikes_a=int(ca.sum()), spikes_b=int(cb.sum()))
    print(f'{label:44s} corr {corr:.3f}  jac {jac:.3f}  ratio {ratio:.3f}  '
          f'prec {prec:.3f}  active {row["active_a"]}/{row["active_b"]}  '
          f'spikes {row["spikes_a"]}/{row["spikes_b"]}')
    return row


# ---------------------------------------------------------------- drift
def drift_check(weights, n, exc, rate):
    """sugar 10 s: rate corr in successive 1 s windows (fast dt 0.25 vs ref)."""
    cfg = DRIFT_CFG
    print(f'\n=== drift check: {cfg["exp"]} {cfg["t_sec"]}s, '
          f'fast dt={cfg["dt"]}, {cfg["window_sec"]}s windows ===')
    t0 = perf_counter()
    ref = run_reference(weights, n, exc, rate, cfg['t_sec'], seed=cfg['seed'],
                        record_times=False, window_sec=cfg['window_sec'])
    print(f'reference 10s done ({perf_counter()-t0:.0f}s)')
    t0 = perf_counter()
    fast = run_fast(exc, rate, cfg['t_sec'], cfg['dt'], seed=cfg['seed'],
                    record_times=False, window_sec=cfg['window_sec'])
    print(f'fast 10s done ({perf_counter()-t0:.0f}s)  overflow={fast["overflow"]}')
    rows = []
    n_win = min(len(ref['windows']), len(fast['windows']))
    for w in range(n_win):
        ca, cb = fast['windows'][w], ref['windows'][w]
        act = (ca > 0) | (cb > 0)
        corr = float(np.corrcoef(ca[act], cb[act])[0, 1]) if act.sum() > 2 else float('nan')
        ratio = float(ca.sum() / max(1, cb.sum()))
        row = dict(kind='drift', exp=cfg['exp'], duration_s=cfg['t_sec'],
                   window=w, dt_fast=cfg['dt'], seed=cfg['seed'],
                   rate_corr=corr, count_ratio=ratio,
                   spikes_a=int(ca.sum()), spikes_b=int(cb.sum()),
                   active_a=int((ca > 0).sum()), active_b=int((cb > 0).sum()))
        rows.append(row)
        print(f'window {w:2d} ({w}-{w+1}s)  corr {corr:.3f}  ratio {ratio:.3f}  '
              f'spikes {row["spikes_a"]}/{row["spikes_b"]}')
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='tiny smoke config')
    mode.add_argument('--full', action='store_true', help='whole matrix (long)')
    ap.add_argument('--exps', nargs='+', default=None, choices=['sugar', 'p9'])
    ap.add_argument('--durations', type=float, nargs='+', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--dts', type=float, nargs='+', default=None)
    ap.add_argument('--no-drift', action='store_true')
    ap.add_argument('--force', action='store_true', help='run even if live_server is up')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit('CUDA not available; this harness benchmarks GPU engines.')

    hits = find_live_server()
    if hits and not args.force:
        for pid, cmd in hits:
            print(f'live_server.py running (pid {pid}): {cmd}')
        sys.exit('ABORT: live_server contention doubles GPU timings (NOTES.md). '
                 'Stop it and rerun, or pass --force.')

    cfg = dict(QUICK if args.quick or not args.full else FULL)
    for key in ('exps', 'durations', 'seeds', 'dts'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_drift:
        cfg['drift'] = False
    print(f'config: {cfg}')

    f2i, _ = get_hash_tables(str(path_comp))
    t0 = perf_counter()
    weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to('cuda')
    n = weights.shape[0]
    print(f'weights loaded: {n} neurons ({perf_counter()-t0:.1f}s)')

    rows = []
    floor_seed = cfg['seeds'][0]
    extra_seeds = [] if len(cfg['seeds']) >= 2 else [floor_seed + 1000]
    for exp_name in cfg['exps']:
        exp = get_experiment(exp_name)
        exc = [f2i[i] for i in exp['neu_exc']]
        rate = exp['stim_rate']
        for dur in cfg['durations']:
            print(f'\n=== {exp_name} {dur}s (rate {rate} Hz) ===')
            refs = {}
            for s in list(cfg['seeds']) + extra_seeds:
                t0 = perf_counter()
                refs[s] = run_reference(weights, n, exc, rate, dur, seed=s)
                print(f'reference seed {s} done ({perf_counter()-t0:.1f}s)')
            for s in list(cfg['seeds'])[1:] + extra_seeds:
                row = compare(refs[s], refs[floor_seed], exc,
                              f'ref-floor seed {s} vs {floor_seed}')
                rows.append(dict(kind='ref-floor', exp=exp_name, duration_s=dur,
                                 seed=s, ref_seed=floor_seed, dt_fast=None, **row))
            for dt in cfg['dts']:
                for s in cfg['seeds']:
                    t0 = perf_counter()
                    fast = run_fast(exc, rate, dur, dt, seed=s)
                    note = f'({perf_counter()-t0:.1f}s, overflow={fast["overflow"]})'
                    row = compare(fast, refs[s], exc, f'fast dt={dt} seed {s} matched {note}')
                    rows.append(dict(kind='fast-matched', exp=exp_name, duration_s=dur,
                                     seed=s, ref_seed=s, dt_fast=dt, **row))
                    if s != floor_seed:
                        row = compare(fast, refs[floor_seed], exc,
                                      f'fast dt={dt} seed {s} vs ref {floor_seed}')
                        rows.append(dict(kind='fast-fixed', exp=exp_name, duration_s=dur,
                                         seed=s, ref_seed=floor_seed, dt_fast=dt, **row))
                    del fast
                    torch.cuda.empty_cache()

    if cfg['drift']:
        exp = get_experiment(DRIFT_CFG['exp'])
        exc = [f2i[i] for i in exp['neu_exc']]
        rows.extend(drift_check(weights, n, exc, exp['stim_rate']))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f'validate_matrix_{stamp}.csv'
    json_path = OUT_DIR / f'validate_matrix_{stamp}.json'
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(
        dict(config={k: v for k, v in cfg.items()}, drift_cfg=DRIFT_CFG,
             precision_window_ms=PRECISION_WINDOW_MS, rows=rows), indent=2))
    print(f'\nsaved {csv_path}\nsaved {json_path}')

    cols = ['kind', 'exp', 'duration_s', 'dt_fast', 'seed',
            'rate_corr', 'jaccard', 'count_ratio', 'spike_precision']
    print('\nsummary:')
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
