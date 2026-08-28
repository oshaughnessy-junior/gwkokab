# Copyright (c) The GWKokab authors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-safe NUTS for long cluster runs that may be evicted or rebooted.

Population inference on rough, latency-bound likelihoods can run for many hours;
on a preemptible scheduler a single eviction or node reboot otherwise discards all
NUTS progress, since the sampler only returns at the end. This helper makes such
runs resumable:

  * run warmup ONCE, then checkpoint the adapted sampler state (mass matrix +
    step size + position + RNG) -- the expensive, fragile part;
  * draw samples in CHUNKS of ``chunk``, atomically checkpointing the sampler
    state together with every sample collected so far after each chunk;
  * on (re)start, resume from the newest checkpoint and continue.

Because the RNG is threaded through ``last_state``, a resumed run reproduces an
uninterrupted run bit-for-bit. Pair with a scheduler policy that returns the
checkpoint directory to the submit side on eviction and restores it on restart
(e.g. HTCondor ``when_to_transfer_output = ON_EXIT_OR_EVICT``).

Example
-------
.. code-block:: python

    from gwkokab.inference import run_nuts_with_checkpoints

    post = run_nuts_with_checkpoints(
        model,
        (T_obs,),
        num_warmup=300,
        num_samples=600,
        key=jax.random.PRNGKey(0),
        ckpt_dir="ckpt",
        chunk=50,
        nuts_kwargs=dict(max_tree_depth=7, target_accept_prob=0.7),
    )
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, Optional, Sequence

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS


class CheckpointExit(Exception):
    """Raised (when ``max_wall_s`` is set) after a checkpoint is safely written and the
    process wall-clock budget is exhausted, to request an exit-driven restart.

    The caller catches this and exits with the HTCondor ``checkpoint_exit_code`` so the
    scheduler spools the checkpoint directory and re-runs the job, which then resumes
    from the newest ``nuts_ckpt.pkl``. Unlike relying on ``ON_EXIT_OR_EVICT`` to
    transfer output on a (possibly ungraceful) eviction, this guarantees the checkpoint
    is preserved across every restart.
    """


def _atomic_pickle(obj: Any, path: str) -> None:
    """Write ``obj`` to ``path`` atomically (a kill mid-write cannot corrupt it)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def run_nuts_with_checkpoints(
    model,
    run_args: Sequence[Any],
    *,
    num_warmup: int,
    num_samples: int,
    key,
    ckpt_dir: str,
    chunk: int = 50,
    nuts_kwargs: Optional[Dict[str, Any]] = None,
    progress: bool = True,
    max_wall_s: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Run NUTS resumably, checkpointing after warmup and after every ``chunk`` samples.

    Parameters
    ----------
    model
        A NumPyro model callable.
    run_args
        Positional arguments passed to the model (e.g. ``(T_obs,)``).
    num_warmup, num_samples
        Warmup and post-warmup sample counts (total, across all resumes).
    key
        PRNG key for warmup.
    ckpt_dir
        Directory holding ``nuts_ckpt.pkl``; created if absent. Resumed from if present.
    chunk
        Samples per checkpoint; smaller is more eviction-resilient, with more overhead.
    nuts_kwargs
        Extra keyword arguments forwarded to :class:`numpyro.infer.NUTS`
        (e.g. ``init_strategy``, ``max_tree_depth``, ``target_accept_prob``).
    progress
        Whether to show the per-chunk progress bar.
    max_wall_s
        If set, the process wall-clock budget in seconds for exit-driven
        checkpointing. After any checkpoint (post-warmup or post-chunk) that leaves
        work remaining, if the elapsed wall time exceeds this budget the function
        raises :class:`CheckpointExit` (the checkpoint is already on disk). The
        caller is expected to translate that into ``sys.exit(checkpoint_exit_code)``
        so HTCondor spools the checkpoint and restarts, resuming from it. ``None``
        (default) keeps the original behavior: run to completion in one process.

    Returns
    -------
    dict
        Posterior samples, one ``(num_samples, ...)`` array per site.

    Raises
    ------
    CheckpointExit
        Only when ``max_wall_s`` is set and the budget is exhausted with work
        remaining; a valid checkpoint has been written before it is raised.
    """
    nuts_kwargs = nuts_kwargs or {}
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "nuts_ckpt.pkl")
    t0 = time.time()

    def _budget_exhausted() -> bool:
        return max_wall_s is not None and (time.time() - t0) > max_wall_s

    state, merged, n_done, phase = None, None, 0, "warmup"
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            d = pickle.load(f)
        phase, state, n_done, merged = d["phase"], d["state"], d["n_done"], d["samples"]
        print(f"[ckpt] RESUME phase={phase} n_done={n_done}/{num_samples}", flush=True)

    # warmup once -> checkpoint the adapted state
    if phase == "warmup":
        m = MCMC(
            NUTS(model, **nuts_kwargs),
            num_warmup=num_warmup,
            num_samples=chunk,
            progress_bar=progress,
        )
        m.warmup(key, *run_args)
        state = jax.device_get(m.last_state)
        _atomic_pickle(
            {"phase": "sampling", "state": state, "n_done": 0, "samples": None}, ckpt
        )
        print(
            "[ckpt] warmup done + checkpointed "
            f"(step_size={float(m.last_state.adapt_state.step_size):.3g})",
            flush=True,
        )
        if _budget_exhausted() and num_samples > 0:
            print(
                f"[ckpt] wall budget {max_wall_s}s hit after warmup -> exit for restart",
                flush=True,
            )
            raise CheckpointExit()

    # sample in chunks, checkpointing after each
    acc = {k: [v] for k, v in merged.items()} if merged else None
    cur = state
    while n_done < num_samples:
        this = min(chunk, num_samples - n_done)
        m = MCMC(
            NUTS(model, **nuts_kwargs),
            num_warmup=0,
            num_samples=this,
            progress_bar=progress,
        )
        m.post_warmup_state = cur
        m.run(cur.rng_key, *run_args)
        s = {k: np.asarray(v) for k, v in m.get_samples().items()}
        if acc is None:
            acc = {k: [v] for k, v in s.items()}
        else:
            for k in acc:
                acc[k].append(s[k])
        n_done += this
        cur = jax.device_get(m.last_state)
        merged = {k: np.concatenate(v) for k, v in acc.items()}
        _atomic_pickle(
            {"phase": "sampling", "state": cur, "n_done": n_done, "samples": merged},
            ckpt,
        )
        print(f"[ckpt] sampled {n_done}/{num_samples}", flush=True)
        if n_done < num_samples and _budget_exhausted():
            print(
                f"[ckpt] wall budget {max_wall_s}s hit at {n_done}/{num_samples} -> exit for restart",
                flush=True,
            )
            raise CheckpointExit()

    return {k: np.concatenate(v) for k, v in acc.items()}
