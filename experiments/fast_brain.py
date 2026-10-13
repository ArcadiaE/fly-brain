"""
FastBrain: event-driven, fixed-shape, CUDA-graph whole-brain LIF stepper.

Same dynamics as code/run_pytorch.py (Shiu et al. LIF + alpha synapse +
delay + refractory + Poisson drive), but

  * recurrent input is computed only from the neurons that spiked
    (row gather from W^T instead of a full sparse mat-vec),
  * every tensor has a fixed shape, so a block of S steps is captured once
    into a CUDA graph and replayed with a single launch,
  * the delay line is a ring buffer whose slots are compile-time constants
    (S is a multiple of the ring length), so no index arithmetic per step,
  * W^T is stored as fixed-width "chunks" (each neuron's outgoing synapses
    split into pieces of chunk_w entries, zero-padded), so the gather needs
    no masks or clamps,
  * the list of active chunks of every step is kept, which is also the list
    of spiking neurons (mapped on the host) -- no extra export pass.

Fixed-shape compaction: pos = cumsum(mask); slot = pos-1 where mask else K;
slot_buf.scatter_(slot, ids). Everything past K lands in a trash slot and is
counted as overflow.

Kernel budget per step is ~33 launches; on a Windows/WDDM laptop GPU each
launch costs 2-10 us of GPU time, hence ~150 us/step regardless of dt.
Use dt=0.25 for realtime.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import pyarrow  # noqa: F401,E402
from run_pytorch import MODEL_PARAMS, get_weights  # noqa: E402
from benchmark import path_comp, path_con, path_wt  # noqa: E402

N_PROBE = 16


class FastBrain:
    def __init__(self, device='cuda', dt=0.1, steps_per_graph=None, params=None,
                 k_chunks=1024, chunk_w=256, use_graph=True, seed=0):
        self.dev = torch.device(device)
        self.dt = float(dt)
        self.p = dict(MODEL_PARAMS if params is None else params)
        self.K, self.W = int(k_chunks), int(chunk_w)
        self.use_graph = use_graph and self.dev.type == 'cuda'
        self.steps_delay = max(1, int(round(self.p['tDelay'] / dt)))
        self.D = self.steps_delay + 1
        # S must be a multiple of the ring length so ring slots are static
        m = max(1, int(round((steps_per_graph or 2 * self.D) / self.D)))
        self.S = m * self.D
        torch.manual_seed(seed)
        self._build_weights()
        self._build_state()
        self.graph = None
        self.t_step = 0

    # ------------------------------------------------------------ weights
    def _build_weights(self):
        W = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True)   # [post, pre]
        n = W.shape[0]
        WT = W.to_sparse_coo().coalesce().t().coalesce().to_sparse_csr()          # rows = pre
        crow = WT.crow_indices()
        col = WT.col_indices().to(torch.int64)
        val = WT.values().float()
        self.n = n
        deg = crow[1:] - crow[:-1]
        nch = torch.div(deg + self.W - 1, self.W, rounding_mode='floor')
        owner = torch.repeat_interleave(torch.arange(n), nch)
        C = len(owner)
        k_in = torch.arange(C) - torch.repeat_interleave(torch.cumsum(nch, 0) - nch, nch)
        start = crow[owner] + k_in * self.W
        length = torch.minimum(deg[owner] - k_in * self.W, torch.full_like(start, self.W))
        # padded chunk tables: real chunks [0, C), dummy chunks [C, C+K)
        T = self.K * self.W                       # trash region size
        pcol = torch.empty(C + self.K, self.W, dtype=torch.int64)
        pval = torch.zeros(C + self.K, self.W, dtype=torch.float32)
        ar = torch.arange(self.W)
        # trash addresses: unique per (chunk, w) modulo T, offset by n
        pcol[:] = n + (torch.arange(C + self.K)[:, None] * self.W + ar[None, :]) % T
        idx = start[:, None] + ar[None, :]
        m = ar[None, :] < length[:, None]
        idx = idx.clamp(max=len(col) - 1)
        pcol[:C][m] = col[idx][m]
        pval[:C][m] = val[idx][m]
        self.C, self.T = C, T
        self.pcol = pcol.to(self.dev)
        self.pval = pval.to(self.dev)
        self.ch_owner = owner.to(self.dev)
        self.ch_owner_cpu = owner.numpy()
        self.ar_w = torch.arange(self.W, device=self.dev)
        self.ar_c = torch.arange(C, device=self.dev)
        self.dummy = (C + torch.arange(self.K + 1, device=self.dev)).clamp(max=C + self.K - 1)
        self.deg_max = int(deg.max())

    # ------------------------------------------------------------ state
    def _build_state(self):
        n, p, dt, dev = self.n, self.p, self.dt, self.dev
        self.base_refrac = float(int(round(p['tRefrac'] / dt)))
        self.refrac_steps = torch.full((n,), self.base_refrac, device=dev)
        self.rates = torch.zeros(n, device=dev)
        self.p_rate = torch.zeros(n, device=dev)         # rates * dt/1000
        self.vth = torch.full((n,), p['vThreshold'], device=dev)   # +inf = silenced
        self.v = torch.full((n,), p['v0'], device=dev)
        self.g = torch.zeros(n, device=dev)
        self.spk = torch.zeros(n, dtype=torch.bool, device=dev)
        self.refrac = self.refrac_steps.clone()
        self.ring = torch.zeros(self.D, n, device=dev)
        self.rec = torch.zeros(n + self.T, device=dev)
        self.slot_buf = torch.empty(self.K + 1, dtype=torch.int64, device=dev)
        # per-block outputs
        self.act_buf = torch.empty(self.S, self.K, dtype=torch.int64, device=dev)
        self.nact_buf = torch.empty(self.S, dtype=torch.int64, device=dev)
        self.probe_idx = torch.zeros(N_PROBE, dtype=torch.int64, device=dev)
        self.probe_v = torch.zeros(self.S, N_PROBE, device=dev)
        self.n_probe = 0
        # constants
        self.f_syn = 1.0 - dt / p['tauSyn']
        self.f_mem = dt / p['tauMem']
        self.w_scale = p['wScale']
        self.poi_amp = p['scalePoisson'] * p['wScale']
        self.v_rest = p['vRest']
        self.v_reset = p['vReset']
        self.v_reset_t = torch.tensor(p['vReset'], device=dev)
        self.zero_t = torch.tensor(0.0, device=dev)

    # ------------------------------------------------------------ controls
    def _idx(self, idx):
        return torch.as_tensor(list(idx), dtype=torch.int64, device=self.dev)

    def set_rate(self, idx, rate):
        idx = self._idx(idx)
        if idx.numel() == 0:
            return
        self.rates[idx] = float(rate)
        self.p_rate[idx] = float(rate) * self.dt / 1000.0
        self.refrac_steps[idx] = 0.0 if rate > 0 else self.base_refrac

    def clear_rates(self):
        self.rates.zero_(); self.p_rate.zero_()
        self.refrac_steps.fill_(self.base_refrac)

    def set_alive(self, idx, alive):
        idx = self._idx(idx)
        if idx.numel():
            self.vth[idx] = self.p['vThreshold'] if alive else float('inf')

    def silenced(self):
        return torch.isinf(self.vth).nonzero(as_tuple=True)[0].tolist()

    def set_probe(self, idx):
        idx = list(idx)[:N_PROBE]
        self.n_probe = len(idx)
        pad = idx + [0] * (N_PROBE - len(idx))
        self.probe_idx.copy_(torch.as_tensor(pad, dtype=torch.int64))

    def reset_state(self):
        self.v.fill_(self.p['v0']); self.g.zero_(); self.spk.zero_()
        self.refrac.copy_(self.refrac_steps); self.ring.zero_()
        self.t_step = 0

    def stimulated(self):
        return (self.rates > 0).nonzero(as_tuple=True)[0].tolist()

    # ------------------------------------------------------------ one step
    def _step(self, s):
        n, K, W = self.n, self.K, self.W
        wslot = s % self.D                       # static ring slots (S % D == 0)
        rslot = (s + 1) % self.D
        # ---- event-driven recurrent input from last step's spikes ----
        cs = self.spk[self.ch_owner]                                   # bool (C)
        pos = torch.cumsum(cs, 0)                                      # int64 (C)
        slot = torch.where(cs, (pos - 1).clamp_(max=K), K)
        self.slot_buf.copy_(self.dummy)
        self.slot_buf.scatter_(0, slot, self.ar_c)
        act = self.slot_buf[:K]                                        # chunk ids (real or dummy)
        cols = self.pcol[act].reshape(-1)
        vals = self.pval[act].reshape(-1)
        self.rec.zero_().index_add_(0, cols, vals)
        self.act_buf[s].copy_(act)
        self.nact_buf[s].copy_(pos[-1])
        # ---- delay ring ----
        torch.mul(self.rec[:n], self.w_scale, out=self.ring[wslot])
        delayed = self.ring[rslot]
        # ---- refractory / synapse / neuron (same order as run_pytorch) ----
        refrac = torch.where(self.spk, 0.0, self.refrac + 1.0)
        ok = (refrac >= self.refrac_steps).float()
        g_new = torch.addcmul(self.g * self.f_syn, delayed, ok)
        u = torch.rand(n, device=self.dev)
        v_in = torch.where(u < self.p_rate, self.v + self.poi_amp, self.v)
        v_new = torch.lerp(v_in, self.g + self.v_rest, self.f_mem)
        spk = v_new > self.vth
        # ---- commit ----
        torch.where(spk, self.v_reset_t, v_new, out=self.v)
        torch.where(spk, self.zero_t, g_new, out=self.g)
        self.refrac.copy_(refrac)
        self.spk.copy_(spk)
        # always record: with CUDA graphs this line must be captured even when
        # no probe is set yet, otherwise later set_probe() has no effect
        self.probe_v[s].copy_(v_new[self.probe_idx])

    def _block(self):
        for s in range(self.S):
            self._step(s)

    # ------------------------------------------------------------ run
    def _capture(self):
        snap = self._snapshot()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._block()
        torch.cuda.current_stream().wait_stream(st)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._block()
        self._restore(snap)

    def _snapshot(self):
        return [x.clone() for x in (self.v, self.g, self.spk, self.refrac, self.ring)]

    def _restore(self, snap):
        for dst, src in zip((self.v, self.g, self.spk, self.refrac, self.ring), snap):
            dst.copy_(src)

    def run_block(self):
        """Advance S steps (one graph replay)."""
        if self.use_graph:
            if self.graph is None:
                self._capture()
            self.graph.replay()
        else:
            with torch.no_grad():
                self._block()
        self.t_step += self.S

    def block_spikes_host(self):
        """Spikes of the last block on the host.

        Returns (list of per-step neuron-id arrays, overflow_chunks).
        Step s of the block holds the spikes emitted at step s-1 (the gather
        runs at the start of a step), so the list is shifted by one step.
        """
        act = self.act_buf.cpu().numpy()
        nact = self.nact_buf.cpu().numpy()
        out = []
        for s in range(self.S):
            a = act[s]
            a = a[a < self.C]
            ids = np.unique(self.ch_owner_cpu[a]) if len(a) else np.zeros(0, np.int64)
            out.append(ids)
        overflow = int(np.clip(nact - self.K, 0, None).sum())
        return out, overflow

    def block_probe_host(self):
        return self.probe_v[:, :self.n_probe].cpu().numpy()


# ---------------------------------------------------------------- self-test
if __name__ == '__main__':
    import argparse
    from time import perf_counter
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--S', type=int, default=None)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--exp', default='sugar')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    from benchmark import get_experiment
    from run_pytorch import get_hash_tables
    exp = get_experiment(args.exp)
    flyid2i, i2flyid = get_hash_tables(str(path_comp))
    exc = [flyid2i[i] for i in exp['neu_exc']]

    fb = FastBrain(dt=args.dt, steps_per_graph=args.S, use_graph=not args.no_graph, seed=args.seed)
    fb.set_rate(exc, exp['stim_rate'])
    print(f'n={fb.n} chunks={fb.C} deg_max={fb.deg_max} dt={fb.dt} S={fb.S} graph={fb.use_graph} '
          f'gpu_mem={torch.cuda.memory_allocated()/2**30:.2f} GB')
    blocks = int(round(args.t * 1000 / args.dt / fb.S))
    fb.run_block(); torch.cuda.synchronize()
    fb.reset_state()
    counts = np.zeros(fb.n, np.int64)
    overflow = 0
    t0 = perf_counter()
    for _ in range(blocks):
        fb.run_block()
        sp, of = fb.block_spikes_host()
        overflow += of
        for ids in sp:
            counts[ids] += 1
    torch.cuda.synchronize()
    wall = perf_counter() - t0
    steps = blocks * fb.S
    print(f'{steps} steps in {wall:.2f}s  ->  {wall/steps*1e6:.0f} us/step   '
          f'{steps*args.dt/1000/wall:.2f}x realtime   overflow_chunks={overflow}')
    print(f'active {(counts>0).sum()}  spikes {counts.sum()}  '
          f'stim mean rate {counts[exc].mean()/args.t:.0f} Hz')
    out = ROOT / 'experiments/out'; out.mkdir(exist_ok=True)
    np.save(out / f'fast_counts_{args.exp}_dt{args.dt}_s{args.seed}.npy', counts)
