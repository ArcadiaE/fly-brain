"""
FastBrain benchmark runner for the Drosophila brain model.

Wraps the event-driven CUDA-Graph engine in experiments/fast_brain.py
(same LIF + alpha-synapse dynamics as run_pytorch.py).  FastBrain is
single-trial and a captured CUDA graph owns its RNG state, so n_run is
implemented as sequential independent trials, each with a fresh FastBrain
instance (weight build + graph capture) and a different seed — the same
rebuild-per-trial convention as the Brian2GeNN runner.

Called by benchmark.py orchestrator.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))

import pandas as pd
import pyarrow  # noqa: F401  — must be imported before torch to avoid libarrow conflict
import numpy as np
import torch
from time import perf_counter as time
import traceback

from benchmark import (
    T_RUN_VALUES_SEC, N_RUN_VALUES,
    path_comp,
    get_experiment, get_spike_output_path, print_summary_table, save_result_csv,
    spike_io_enabled,
)
from run_pytorch import get_hash_tables
from fast_brain import FastBrain

# Validated realtime setting (see experiments/NOTES.md); 0.1 also validated.
DEFAULT_DT_VALUES = [0.25]

BASE_SEED = 1


def _fmt_dt(dt):
    return f'{dt:g}'


def run_single_benchmark(t_run_sec, n_run, experiment, logger, dt,
                         run_idx=None, total_runs=None,
                         run_label=None, round_idx=None):
    """
    Run a single FastBrain benchmark with specified t_run, n_run and dt.

    n_run independent trials run sequentially; each trial builds a fresh
    FastBrain instance (CUDA graph capture owns RNG state) with seed
    BASE_SEED + trial_idx (+ round offset), mirroring the Brian2GeNN
    rebuild-per-trial convention.
    """
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend_key = f'fastbrain_dt{_fmt_dt(dt)}'
    exp_name = f'{backend_key}_t{t_run_sec}s_n{n_run}'

    run_info = f"[{run_idx}/{total_runs}] " if run_idx else ""
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"{run_info}BENCHMARK: t_run={t_run_sec}s, n_run={n_run}, dt={dt}ms")
    logger.log_raw("=" * 80)
    logger.log(f"Device: {device_name.upper()}")
    logger.log(f"Experiment: {exp_name}")
    record_spikes = spike_io_enabled()
    logger.log(
        "Spike probing/output: "
        f"{'enabled' if record_spikes else 'disabled'}"
    )
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")

    timings = {
        'id_mapping': 0.0,
        'weight_build': 0.0,      # weight load + chunk tables + state
        'device_build': 0.0,      # CUDA graph capture (warmup block)
        'simulation_total': 0.0,  # graph replays only
    }

    try:
        # ===== ID mappings =====
        t_mapping_start = time()
        flyid2i, i2flyid = get_hash_tables(str(path_comp))
        exc_indices = [flyid2i[n] for n in experiment['neu_exc']]
        timings['id_mapping'] = time() - t_mapping_start
        logger.log(f"ID mapping:         {timings['id_mapping']:.3f}s")

        stim_rate = experiment['stim_rate']
        all_times_ms = []
        all_neurons = []
        all_trials = []
        overflow_total = 0

        logger.log(
            f"Running {n_run} independent trial(s); each trial builds a "
            "fresh FastBrain instance (graph capture owns RNG state)."
        )
        for trial_idx in range(n_run):
            trial_label = f"{trial_idx + 1}/{n_run}"
            trial_seed = BASE_SEED + trial_idx
            if round_idx is not None:
                trial_seed += int(round_idx) * 100000

            # ===== Per-trial setup: weights + state =====
            t_setup_start = time()
            fb = FastBrain(device=device_name, dt=dt, seed=trial_seed)
            timings['weight_build'] += time() - t_setup_start

            # ===== Per-trial build: CUDA graph capture =====
            t_capture_start = time()
            fb.run_block()  # first call triggers graph capture
            if device_name == 'cuda':
                torch.cuda.synchronize()
            timings['device_build'] += time() - t_capture_start
            fb.reset_state()
            fb.set_rate(exc_indices, stim_rate)

            # ===== Per-trial simulation (graph replays only) =====
            steps = int(round(t_run_sec * 1000.0 / dt))
            blocks = max(1, int(round(steps / fb.S)))
            t_actual = blocks * fb.S * dt / 1000.0
            block_act = []
            block_nact = []
            t_sim_start = time()
            for b in range(blocks):
                fb.run_block()
                if record_spikes:
                    block_act.append(fb.act_buf.clone())
                    block_nact.append(fb.nact_buf.clone())
            if device_name == 'cuda':
                torch.cuda.synchronize()
            timings['simulation_total'] += time() - t_sim_start

            if trial_idx == 0:
                logger.log(f"  Neurons: {fb.n}, S={fb.S} steps/block")
                logger.log(
                    f"  Steps: {steps} requested, {blocks * fb.S} actual "
                    f"({blocks} blocks, t_actual={t_actual:.4f}s)"
                )
                if device_name == 'cuda':
                    free, total = torch.cuda.mem_get_info(device_name)
                    logger.log(f"  VRAM after setup: {(total - free) / 1024 ** 3:.2f} GB")

            # ===== Per-trial spike extraction (outside timed sim) =====
            if record_spikes:
                act_cpu = [a.cpu().numpy() for a in block_act]
                nact_cpu = [na.cpu().numpy() for na in block_nact]
                del block_act, block_nact
                C, S, K = fb.C, fb.S, fb.K
                owner = fb.ch_owner_cpu
                for b, (act, nact) in enumerate(zip(act_cpu, nact_cpu)):
                    overflow_total += int(np.clip(nact - K, 0, None).sum())
                    for s in range(S):
                        a = act[s]
                        a = a[a < C]
                        if len(a):
                            ids = np.unique(owner[a])
                            all_times_ms.append(
                                np.full(len(ids), (b * S + s) * dt)
                            )
                            all_neurons.append(ids)
                            all_trials.append(
                                np.full(len(ids), trial_idx, dtype=np.int64)
                            )

            logger.log(
                f"  Trial {trial_label}: seed {trial_seed}"
            )
            del fb
            if device_name == 'cuda':
                torch.cuda.empty_cache()

        timings['model_setup_total'] = timings['weight_build']
        timings['simulation_avg_per_trial'] = timings['simulation_total'] / n_run
        logger.log(f"  Total weight build: {timings['weight_build']:.3f}s")
        logger.log(f"  Total graph capture: {timings['device_build']:.3f}s")
        logger.log(f"  Total simulation:   {timings['simulation_total']:.3f}s")
        logger.log(f"  Avg per trial:      {timings['simulation_avg_per_trial']:.3f}s")
        if overflow_total:
            logger.log(f"  WARNING: overflow chunks: {overflow_total}")

        # ===== Collect and save results =====
        logger.log("Collecting results...")
        path_save = ''
        if not record_spikes:
            df = pd.DataFrame(
                {
                    't': [], 'time_ms': [], 'trial': [],
                    'neuron_index': [], 'flywire_id': [], 'exp_name': [],
                }
            )
            timings['result_collection'] = 0.0
            timings['result_save'] = 0.0
            logger.log("  Spike probing/output disabled; no parquet written")
        else:
            t_collect_start = time()
            if all_neurons:
                times = np.concatenate(all_times_ms)
                neurons = np.concatenate(all_neurons)
                trials = np.concatenate(all_trials)
            else:
                times = np.zeros(0)
                neurons = np.zeros(0, dtype=np.int64)
                trials = np.zeros(0, dtype=np.int64)
            df = pd.DataFrame({
                't': times,
                'trial': trials,
                'neuron_index': neurons,
                'flywire_id': [i2flyid[int(n)] for n in neurons],
                'exp_name': exp_name,
            })
            df.insert(1, 'time_ms', df['t'])
            timings['result_collection'] = time() - t_collect_start

            t_save_start = time()
            path_save = get_spike_output_path(exp_name, run_label, round_idx)
            path_save.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path_save, compression='brotli')
            timings['result_save'] = time() - t_save_start

        logger.log(f"  Collection:       {timings['result_collection']:.3f}s")
        logger.log(f"  Save to file:     {timings['result_save']:.3f}s")
        if path_save:
            logger.log(f"  Output file:      {path_save}")

        # ===== Totals and metrics =====
        timings['total_elapsed'] = (
            timings['id_mapping']
            + timings['model_setup_total']
            + timings['device_build']
            + timings['simulation_total']
            + timings['result_collection']
            + timings['result_save']
        )

        total_simulated_time = t_run_sec * n_run
        timings['realtime_ratio'] = (
            total_simulated_time / timings['simulation_total']
            if timings['simulation_total'] > 0 else float('inf')
        )
        timings['realtime_ratio_total'] = (
            total_simulated_time / timings['total_elapsed']
            if timings['total_elapsed'] > 0 else float('inf')
        )

        n_active = df['flywire_id'].nunique() if len(df) > 0 else 0
        n_spikes = len(df)

        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': n_active,
            'n_spikes': n_spikes,
            'status': 'success',
            'timings': timings,
            'backend_key': backend_key,
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
            'spike_path': str(path_save),
        }

        # ===== Summary =====
        logger.log_raw("")
        logger.log_raw("-" * 60)
        logger.log("TIMING SUMMARY")
        logger.log_raw("-" * 60)
        logger.log(f"  Weight build:       {timings['weight_build']:>10.3f}s")
        logger.log(f"  Graph capture:      {timings['device_build']:>10.3f}s")
        logger.log(f"  Simulation:         {timings['simulation_total']:>10.3f}s")
        logger.log(f"  Result processing:  {timings['result_collection'] + timings['result_save']:>10.3f}s")
        logger.log("  -----------------------------------------")
        logger.log(f"  TOTAL ELAPSED:      {timings['total_elapsed']:>10.3f}s")
        logger.log_raw("")
        logger.log(f"  Simulated time:     {total_simulated_time:>10.1f}s ({n_run} x {t_run_sec}s)")
        logger.log(f"  Realtime ratio (sim only): {timings['realtime_ratio']:>6.3f}x")
        logger.log(f"  Realtime ratio (total):    {timings['realtime_ratio_total']:>6.3f}x")
        logger.log_raw("")
        logger.log(f"  Active neurons:     {n_active:>10d}")
        logger.log(f"  Total spikes:       {n_spikes:>10d}")
        logger.log_raw("-" * 60)

    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        logger.log_raw(traceback.format_exc())
        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': 0,
            'n_spikes': 0,
            'status': f'error: {str(e)}',
            'timings': timings,
            'backend_key': backend_key,
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
        }

    return results


def run_all_benchmarks(t_run_values=None, n_run_values=None,
                       experiment=None, logger=None,
                       run_label=None, round_idx=None, dt_values=None):
    """
    Run all FastBrain benchmark combinations (dt x n_run x t_run).

    Args:
        t_run_values: List of t_run durations in seconds, or None for all
        n_run_values: List of n_run values to test, or None for all
        experiment: experiment config dict from get_experiment()
        logger: BenchmarkLogger instance
        dt_values: List of timesteps in ms, or None for DEFAULT_DT_VALUES
    """
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES
    if dt_values is None:
        dt_values = DEFAULT_DT_VALUES
    if experiment is None:
        experiment = get_experiment()

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name != 'cuda':
        logger.log(
            "WARNING: CUDA not available; FastBrain falls back to "
            "CPU without CUDA graphs (not representative)."
        )

    all_results = []

    for dt in dt_values:
        backend_name = f'FastBrain dt={_fmt_dt(dt)}ms (CUDA Graph)'

        benchmarks = []
        for n_run in n_run_values:
            for t_run_sec in t_run_values:
                benchmarks.append((t_run_sec, n_run))

        total_runs = len(benchmarks)

        logger.log_raw("")
        logger.log_raw("=" * 80)
        logger.log(f"BENCHMARK SUITE: {backend_name}")
        logger.log_raw("=" * 80)
        logger.log(f"Device: {device_name.upper()}")
        if device_name == 'cuda':
            logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(f"dt: {dt} ms")
        logger.log(f"t_run values: {t_run_values} seconds")
        logger.log(f"n_run values: {n_run_values}")
        if run_label:
            logger.log(f"Run label: {run_label}")
        if round_idx is not None:
            logger.log(f"Round: {round_idx}")
        logger.log(f"Total benchmarks: {total_runs}")
        logger.log_raw("=" * 80)

        dt_results = []
        for run_idx, (t_run_sec, n_run) in enumerate(benchmarks, 1):
            result = run_single_benchmark(
                t_run_sec=t_run_sec,
                n_run=n_run,
                experiment=experiment,
                logger=logger,
                dt=dt,
                run_idx=run_idx,
                total_runs=total_runs,
                run_label=run_label,
                round_idx=round_idx,
            )
            dt_results.append(result)
            save_result_csv(backend_name, result)

        print_summary_table(dt_results, backend_name, logger)
        all_results.extend(dt_results)

    return all_results
