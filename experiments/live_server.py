"""
Live fly-brain console: runs the whole-brain LIF model on the GPU and streams
spikes + membrane voltages to a browser page over WebSocket. The page can
change which neurons are stimulated, at what rate, silence neurons, and pick
which neurons to show on the oscilloscope.

Backend: experiments/fast_brain.py (event-driven + CUDA graph, ~1x realtime
at dt=0.25 ms on an RTX 3060 laptop). --dense uses the original
code/run_pytorch.py model instead (about 0.05x realtime).

Usage (inside .venv):
    python experiments/live_server.py                 # http://127.0.0.1:8766, dt 0.25
    python experiments/live_server.py --dt 0.1        # reference time step, ~0.4x realtime
    python experiments/live_server.py --speed 0       # run as fast as the GPU allows
    python experiments/live_server.py --dense         # original dense model

Frame (server -> client, binary, one per simulation block):
    uint32 step | uint32 n_spikes | uint32 n_probe | uint32 S | float32 us_per_step
    uint32[n_spikes] neuron ids (unique within the block)
    float32[S * n_probe] probe voltages, row-major (step, probe)
Commands (client -> server, JSON): see handle() below.
"""
import argparse
import asyncio
import json
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'experiments'))

import pyarrow  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from aiohttp import web, WSMsgType  # noqa: E402

from benchmark import EXPERIMENTS, path_comp  # noqa: E402
from run_pytorch import MODEL_PARAMS, get_hash_tables  # noqa: E402

ANN_PATH = ROOT / 'data/annotations/Supplemental_file1_neuron_annotations.tsv'
PAGE = ROOT / 'experiments/brain_live.html'
CLASSES = ['optic', 'central', 'sensory', 'visual_projection', 'ascending',
           'descending', 'sensory_ascending', 'visual_centrifugal', 'motor',
           'endocrine', 'unknown']


class Sim:
    """Runs FastBrain (or the dense reference) in a thread and queues frames."""

    def __init__(self, device, dt, speed, dense, steps_per_graph):
        self.dev = device
        self.speed = speed            # 1.0 = realtime cap, 0 = unthrottled
        self.flyid2i, self.i2flyid = get_hash_tables(str(path_comp))
        if dense:
            from run_pytorch import DT
            self.dt = DT
            self.core = DenseCore(device, steps_per_graph or 20)
        else:
            from fast_brain import FastBrain
            self.dt = dt
            self.core = FastBrain(device=device, dt=dt, steps_per_graph=steps_per_graph)
        self.n = self.core.n
        self.S = self.core.S
        self.mode = 'dense' if dense else 'fast'
        self.lock = threading.Lock()
        self.running = True
        self.step = 0
        self.us_per_step = 0.0
        self.counts = np.zeros(self.n, np.int64)
        self.overflow = 0
        self.out = deque(maxlen=256)
        self.probe = []
        self.thread = threading.Thread(target=self.loop, daemon=True)

    # ---- controls (called from the event loop thread) ----
    def set_rate(self, idx, rate):
        with self.lock:
            self.core.set_rate(idx, rate)

    def clear(self):
        with self.lock:
            self.core.clear_rates()

    def set_alive(self, idx, alive):
        with self.lock:
            self.core.set_alive(idx, alive)

    def set_probe(self, idx):
        with self.lock:
            self.probe = list(idx)[:16]
            self.core.set_probe(self.probe)

    def reset(self):
        with self.lock:
            self.core.reset_state()
            self.step = 0
            self.counts[:] = 0
            self.overflow = 0
            self.out.clear()

    def stimulated(self):
        return self.core.stimulated()

    def silenced(self):
        return self.core.silenced()

    # ---- simulation thread ----
    def loop(self):
        block_ms = self.S * self.dt
        next_wall = time.perf_counter()
        while True:
            if not self.running:
                time.sleep(0.02)
                next_wall = time.perf_counter()
                continue
            t0 = time.perf_counter()
            with self.lock:
                self.core.run_block()
                spikes, of = self.core.block_spikes_host()
                pv = self.core.block_probe_host() if self.probe else np.zeros((self.S, 0), np.float32)
            self.us_per_step = (time.perf_counter() - t0) * 1e6 / self.S
            self.step += self.S
            self.overflow += of
            ids = np.unique(np.concatenate(spikes)) if any(len(s) for s in spikes) else np.zeros(0, np.int64)
            for s in spikes:
                self.counts[s] += 1
            hdr = struct.pack('<IIIIf', self.step, len(ids), pv.shape[1], self.S, self.us_per_step)
            self.out.append(hdr + ids.astype(np.uint32).tobytes() + np.ascontiguousarray(pv, np.float32).tobytes())
            if self.speed > 0:
                next_wall += block_ms / 1000.0 / self.speed
                lag = next_wall - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                elif lag < -0.5:
                    next_wall = time.perf_counter()


class DenseCore:
    """Adapter giving the original TorchModel the FastBrain block interface."""

    def __init__(self, device, S):
        from run_pytorch import DT, TorchModel, get_weights
        from benchmark import path_con, path_wt
        self.dev, self.S, self.dt = device, S, DT
        w = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(device)
        self.n = w.shape[0]
        self.model = TorchModel(1, self.n, DT, MODEL_PARAMS, w, exc_indices=None, device=device)
        self.base_refrac = int(round(MODEL_PARAMS['tRefrac'] / DT))
        self.rates = torch.zeros(1, self.n, device=device)
        self.alive = torch.ones(1, self.n, device=device)
        self.state = self.model.state_init()
        self.gen = torch.Generator(device=device).manual_seed(0)
        self.probe_idx = []
        self._spk = []
        self._pv = np.zeros((S, 0), np.float32)

    def set_rate(self, idx, rate):
        self.rates[0, idx] = float(rate)
        self.model.neurons.refrac_steps[idx] = 0 if rate > 0 else self.base_refrac

    def clear_rates(self):
        self.rates.zero_(); self.model.neurons.refrac_steps.fill_(self.base_refrac)

    def set_alive(self, idx, alive):
        self.alive[0, idx] = 1.0 if alive else 0.0

    def silenced(self):
        return (self.alive[0] == 0).nonzero(as_tuple=True)[0].tolist()

    def set_probe(self, idx):
        self.probe_idx = list(idx)

    def reset_state(self):
        self.state = self.model.state_init()

    def stimulated(self):
        return (self.rates[0] > 0).nonzero(as_tuple=True)[0].tolist()

    def run_block(self):
        st = self.state
        acc = torch.zeros(self.n, device=self.dev)
        pv = torch.zeros(self.S, len(self.probe_idx), device=self.dev)
        pi = torch.as_tensor(self.probe_idx, dtype=torch.long, device=self.dev)
        with torch.no_grad():
            for s in range(self.S):
                st = self.model(self.rates, *st, generator=self.gen)
                st = (st[0], st[1], st[2] * self.alive, st[3], st[4])
                acc += st[2][0]
                if len(self.probe_idx):
                    pv[s] = st[3][0, pi]
        self.state = st
        self._spk = [acc.nonzero(as_tuple=True)[0].cpu().numpy()]
        self._pv = pv.cpu().numpy()

    def block_spikes_host(self):
        return self._spk, 0

    def block_probe_host(self):
        return self._pv


def load_annotations(sim):
    ann = pd.read_csv(ANN_PATH, sep='\t', low_memory=False).set_index('root_id')
    ids = np.array([sim.i2flyid[i] for i in range(sim.n)], dtype=np.int64)
    a = ann.reindex(ids)
    xyz = a[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=np.float32) * np.array([4, 4, 40], np.float32) / 1000
    bad = np.isnan(xyz).any(axis=1)
    xyz -= np.nanmean(xyz, axis=0)
    xyz[bad] = 1e6
    sc = a['super_class'].fillna('unknown').map({c: i for i, c in enumerate(CLASSES)})
    sc = sc.fillna(len(CLASSES) - 1).to_numpy(dtype=np.uint8)
    ctype = a['cell_type'].fillna(a['hemibrain_type']).fillna(a['cell_class']).fillna('').astype(str)
    types = sorted(ctype.unique())
    tcode = ctype.map({t: i for i, t in enumerate(types)}).to_numpy(dtype=np.uint16)
    side = a['side'].fillna('').astype(str)
    scode = side.map({'left': 1, 'right': 2, 'center': 3}).fillna(0).to_numpy(dtype=np.uint8)
    ntl = ['', 'acetylcholine', 'gaba', 'glutamate', 'dopamine', 'serotonin', 'octopamine']
    ntcode = a['top_nt'].fillna('').astype(str).map({t: i for i, t in enumerate(ntl)}).fillna(0).to_numpy(dtype=np.uint8)
    table = pd.DataFrame({'type': ctype.values, 'side': side.values})
    return {'pos': xyz.astype(np.float32), 'sc': sc, 'tcode': tcode, 'scode': scode, 'ntcode': ntcode,
            'types': types, 'nts': ntl, 'ids': ids.astype(str).tolist(), 'table': table}


def make_app(sim, ann):
    app = web.Application()
    clients = set()

    def bin_resp(arr):
        return web.Response(body=arr.tobytes(), content_type='application/octet-stream')

    async def meta(_):
        presets = {k: {'name': v['name'], 'rate': v['stim_rate'],
                       'idx': [sim.flyid2i[i] for i in v['neu_exc'] if i in sim.flyid2i]}
                   for k, v in EXPERIMENTS.items()}
        return web.json_response({
            'n': sim.n, 'dt': sim.dt, 'spb': sim.S, 'mode': sim.mode, 'speed': sim.speed,
            'classes': CLASSES, 'types': ann['types'], 'nts': ann['nts'],
            'v_th': MODEL_PARAMS['vThreshold'], 'v_rest': MODEL_PARAMS['vRest'],
            'device': sim.dev, 'presets': presets,
        })

    def state_msg():
        return {'type': 'state', 'stim': sim.stimulated(), 'silenced': sim.silenced(),
                'running': sim.running, 'probe': sim.probe, 'speed': sim.speed}

    def handle(m):
        cmd = m.get('cmd')
        if cmd == 'stim':
            sim.set_rate(m['idx'], m.get('rate', 100))
        elif cmd == 'stim_type':
            t = ann['table']
            sel = (t['type'] == m['type'])
            if m.get('side'):
                sel &= (t['side'] == m['side'])
            idx = np.nonzero(sel.to_numpy())[0].tolist()
            sim.set_rate(idx, m.get('rate', 100))
            return {'type': 'log', 'msg': f'刺激 {m["type"]} {m.get("side", "")}: {len(idx)} 个神经元 @ {m.get("rate", 100)} Hz'}
        elif cmd == 'unstim':
            sim.set_rate(m['idx'], 0)
        elif cmd == 'clear':
            sim.clear()
        elif cmd == 'silence':
            sim.set_alive(m['idx'], False)
        elif cmd == 'unsilence':
            sim.set_alive(m['idx'], True)
        elif cmd == 'probe':
            sim.set_probe(m['idx'])
        elif cmd == 'pause':
            sim.running = False
        elif cmd == 'resume':
            sim.running = True
        elif cmd == 'reset':
            sim.reset()
        elif cmd == 'speed':
            sim.speed = float(m.get('value', 1.0))
        elif cmd == 'search':
            q = m.get('q', '').lower()
            t = ann['table']
            hit = t[t['type'].str.lower().str.contains(q, regex=False) & (t['type'] != '')]
            g = hit.groupby(['type', 'side']).size().reset_index(name='n')
            g = g.sort_values('n', ascending=False).head(40)
            return {'type': 'search', 'hits': g.values.tolist()}
        elif cmd == 'top':
            k = int(m.get('k', 8))
            c = sim.counts.copy()
            c[sim.stimulated()] = 0
            top = np.argsort(-c)[:k]
            return {'type': 'top', 'idx': [int(i) for i in top if c[i] > 0]}
        return None

    async def ws(request):
        sock = web.WebSocketResponse(max_msg_size=0)
        await sock.prepare(request)
        clients.add(sock)
        await sock.send_json(dict(state_msg(), type='hello'))
        try:
            async for msg in sock:
                if msg.type != WSMsgType.TEXT:
                    continue
                reply = handle(json.loads(msg.data))
                await sock.send_json(state_msg())
                if reply:
                    await sock.send_json(reply)
        finally:
            clients.discard(sock)
        return sock

    async def broadcaster(app):
        last_overflow = 0
        while True:
            if sim.out and clients:
                frames = []
                while sim.out:
                    frames.append(sim.out.popleft())
                data = b''.join(frames)
                for c in list(clients):
                    try:
                        await c.send_bytes(data)
                    except Exception:
                        clients.discard(c)
                if sim.overflow != last_overflow:
                    last_overflow = sim.overflow
                    for c in list(clients):
                        await c.send_json({'type': 'log', 'msg': f'警告：活动过高，丢弃了 {sim.overflow} 个突触块', 'cls': 'e'})
            else:
                if not clients:
                    sim.out.clear()
                await asyncio.sleep(0.005)

    async def start_bg(app):
        app['bg'] = asyncio.create_task(broadcaster(app))

    app.on_startup.append(start_bg)
    app.router.add_get('/', lambda r: web.FileResponse(PAGE))
    app.router.add_get('/api/meta', meta)
    app.router.add_get('/api/ids', lambda r: web.json_response(ann['ids']))
    for name in ('pos', 'sc', 'tcode', 'scode', 'ntcode'):
        app.router.add_get(f'/api/{name}.bin', (lambda nm: (lambda r: bin_resp(ann[nm])))(name))
    app.router.add_get('/ws', ws)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--dt', type=float, default=0.25, help='ms; 0.1 = reference, 0.25 = realtime')
    ap.add_argument('--speed', type=float, default=1.0, help='realtime cap (1 = realtime, 0 = max)')
    ap.add_argument('--dense', action='store_true', help='use the original dense model')
    ap.add_argument('--steps', type=int, default=None, help='steps per block/graph')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    print('loading brain ...')
    sim = Sim(args.device, args.dt, args.speed, args.dense, args.steps)
    print(f'  {sim.n} neurons on {args.device}, mode={sim.mode}, dt={sim.dt} ms, {sim.S} steps/block')
    print('loading annotations ...')
    ann = load_annotations(sim)
    print(f'  {len(ann["types"])} cell types')
    sim.core.run_block()          # capture the CUDA graph before serving
    sim.core.reset_state()
    sim.thread.start()
    print(f'open  http://127.0.0.1:{args.port}/')
    web.run_app(make_app(sim, ann), host='127.0.0.1', port=args.port, print=None)


if __name__ == '__main__':
    main()
