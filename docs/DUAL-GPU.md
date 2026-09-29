# DUAL-GPU: a layer split across two cards

**Status: PHASE 1 LANDED AND COMPILES CLEAN** (CUDA 12.0, sm_86, GCC 13, all targets including the test
binaries; one fix the compiler caught: `session_run_token`'s definition needed the new `done` parameter).
The decode-path split is NOT yet wired - `--split-layers` loads both cards and then refuses rather than
decode on a half-split engine. Written for two RTX 3060
12 GB; the point is not parallelism — the 48-layer residual chain is strictly serial (`session.hpp:117`) —
but **capacity**: dense weights do not duplicate, so two cards hold roughly 3x the expert cache of one,
which cuts the exposed CPU expert term (`expert_cache.hpp:3`, the largest single cost in a token).

## What landed (each item leaves the no-flag path byte-identical)

1. **`weights.{hpp,cpp}`**: `index_names()` — the flat index's tensor names, so a side can build its
   "every `blk.<l>.` I do not own" skip set (the loader's compaction does the placement).
2. **`session.{hpp,cpp}`**: `SessionState::layer_begin/layer_end`; `session_bytes_range` /
   `session_init_range` (per-side carve, per-side RoPE, range-local `gdn_point_at`/`stage_token`);
   every capture/replay/loop walks the owning side's range with GLOBAL doorbell ring numbers;
   `session_run_token(..., cudaEvent_t done)` can hand completion to the caller instead of syncing;
   `TokenGraph::layer_begin` keeps the host's ring expectations global across the boundary.
   `layer.cpp`: the doorbell's mapped pages are now `cudaHostAllocPortable` (reachable from both
   devices; no behavior change with one).
3. **`generate.cpp`**: `--split-layers N` (2..47; >= 2 so layer 1's PLE stays on side 0); the `GpuSide`
   scaffold (ordinal, stream, `WeightTable`+arena, `NativeDense`, session arena+state, `parts_dev`,
   `ExpertCache`+slot count+sized slots); per-side weight loading with foreign-layer skip sets
   (`token_embd` -> side 0, `output` -> last side); per-side session init and streams; per-side
   expert-cache auto-sizing from THAT card's free VRAM and profile sliced by layer ownership; the MTP
   drafter, native head and logits on the last side's device. A `DeviceGuard` RAII makes every
   allocation land on its owning card. **A run with `--split-layers` stops after the caches with an
   explicit refusal** naming this file - running the un-split decode path on a half-split engine would
   read side-1 layers from side-0's null tensors, i.e. plausible tokens from the wrong weights.

## What remains, in order, with the seams

4. **`verify.{cpp,hpp}` - LANDED (commit a59f1cd).** Two `Verifier` instances, one per side;
   `set_input_external()` + `adopt_state(prev, T)` carry the boundary (`R_`, `bo_`, `inj2_`); the head/
   sampling/`out` exist only on the last side; `run()`'s pool callback still receives the GLOBAL layer
   number. What remains HERE: nothing inside verify.cpp - the generate.cpp driver (item 8) instantiates
   and orchestrates the pair. `Verifier::init(wt, g, ss, hits,
   head, spec, err)` and `record_window` assume one device. Split: record the window per side into two
   `exec_[T]` tables (capture stream per side), `run()` launches side 0's half, serves its rings/pool/
   flags, then `cudaEventRecord` on side 0's stream + `cudaStreamWaitEvent` on side 1's + a peer copy of
   the window's `R` (T rows x hc x n_embd) + side 1's half. The `VerifyHits` (d_res/cache_base/hit
   buffers) become per side; `emb_` stays side 0; `R_`/sampler/logits on side 1; `commit()` per side.
   The doorbell/flag protocol needs no change - seq is global and the halves are device-serialized.
5. **`prefill.{cpp,hpp}`** (READ: the chunk forward is `for l in 0..n_layers` at `prefill.cpp:855` with
   `gdn_index`/`qsa_index` locals, `ss.gdn_state + gdn_index*gdn_floats`, `ss.qsa_states[qsa_index]`,
   weights via `LayerView(*m.wt, l)`, chunk activations in `m.R` (T rows x hc*n_embd); the embeddings
   broadcast into `m.R` happens just before the loop (~:758)). Split = the same shape as the verifier:
   a layer range on `Prefill` (from `ss.layer_begin/end`), `gdn_index`/`qsa_index` counting IN RANGE,
   the loop over `[begin, end)`, the embedding+broadcast only on side 0, and a boundary handoff of
   `m.R` (T x hc x n_embd) between side 0's last layer and side 1's first - confirm against the MoE/
   residual tail of the loop (~:1035-1200, unread) whether anything else crosses (the verifier needed
   `bo_`/`inj2_` because its FUSED gr path defers the FFN fold; prefill's unfused `gr_norm`/`gr_mix`
   halves look like they fold within the layer). `Prefill::init` gets one instance per side (its own
   `wt`/`ss`/`xcache`/stream/cuBLAS `gemm.init_external`/staging ring - the ring streams only its
   side's experts over its own PCIe link, which is where TTFT improves); `on_chunk` (the MTP prefill
   feed) wires to SIDE 1'S instance only - it must see the residual AFTER side 1's layers; the
   cache-slot borrowing/lending (`lend_slots`/`plan_lend`/`Prefill::relayout`) becomes per side.
   The driver runs side 0's chunk then side 1's chunk per prompt chunk with the handoff between.
6. **`mtp.{cpp,hpp}`**: the drafter is placed on the last side (its `load` is already DeviceGuarded
   there; `bind(wt, ...)` must take the last side's table and the side-1 verifier's `final_R_all()`).
7. **`expert_source.hpp` / `ExpertDispatch`**: a second pointer set (parts/hit/cache) selected by its
   own layer counter when `split_at` is set, or one `ExpertDispatch` per side with the pool callback
   dispatching on the layer it is serving.
8. **`generate.cpp` decode drivers + checkpoints**: the `ConvCheckpoint` save/restore per side (gather/
   scatter), the `--serve` request loop driving the split verify+prefill, `host_res`/`d_res` per side,
   the adapt/swaps refill per side. Then delete the refusal.
9. **`serve/server.py` + `setup.py`**: `"gpus": [a, b]` config; `child_env` sets `CUDA_VISIBLE_DEVICES=a,b`;
   setup writes both; telemetry watches both NVML indices; START-HERE passes `--split-layers 24`.
10. **`tools/bench_dual.py`**: greedy, fixed seed, `--split-layers 0` vs `24`, asserts identical token
    streams, reports tok/s + TTFT + per-card VRAM.

Expected: ~+30-45% output tokens/s at 128K context, a shorter TTFT, and 62 tok/s (2x) unreachable.
"More VRAM matters more than a faster GPU" (`DETAILS.md:92`).

## The contract

`strata generate --split-layers N` runs layers `[0, N)` on visible device 0 and `[N, 48)` on visible device
1 (1 <= N <= 47). Absent the flag, the engine is byte-for-byte today's single-device engine: same calls, same
order, device 0. The server (`serve/server.py`) stops masking `CUDA_VISIBLE_DEVICES` to one card when the
config selects two.

## What crosses the boundary

Exactly one tensor per token (per window row): the gated residual `R`, `(hc, n_embd)` floats = 40 KB/row.
The boundary falls between `post[N-1]` and `pre[N]` — a layer boundary, never mid-MoE, so `parts` never
crosses. The copy is enqueued stream-ordered (event on side 0's stream, wait on side 1's, then a peer copy
that the driver stages through host if the pair has no P2P), NOT a host sync — host calls on the token path
are the measured enemy (`session.hpp:183`).

Everything else is device-local by construction:

* dense weights: per-side `WeightTable` + arena, via the loader's existing `skip` set (compacted arena,
  `resident == false` rows — `weights.cpp:169`); global tensors placed on their only consumer
  (`token_embd.weight` side 0, `output.weight` side 1, `blk.1.ple_*` side 0 with layer 1).
* session state: per-side carve of GDN/QSA state **for its layer range only** (RoPE table duplicated, one
  per side — 64 MiB, acceptable).
* KV: each side's QSA layers own their pools; the streaming host copy is shared (it is per-layer pinned host
  memory, reachable from both devices).
* expert cache: one per side, auto-sized from THAT card's free VRAM, its `d_res` rows filled only for owned
  layers (foreign rows stay `kNotResident`), profile sliced by layer ownership.
* the doorbell is mapped pinned memory with a monotonic `h_seq` that keeps its GLOBAL layer numbering across
  both halves, so the host's ring/flag protocol is unchanged (`cudaHostAllocPortable` on its pages).

## Files and the shape of each change

1. **`src/core/weights.cpp` + `include/strata/core/weights.hpp`** — `index_names(pack, out, err)`: the flat
   index's tensor names, so callers can build "every `blk.<l>.` name with `l` outside my range" skip sets
   without re-parsing the format a third time.
2. **`src/core/session.cpp` + `include/strata/core/session.hpp`** — range-aware session: `session_bytes` /
   `session_init` take `[layer_begin, layer_end)`; `gdn_point_at` and `stage_token` count in-range layers;
   `session_capture_token` / `session_loop` take a side context (device, stream, parts). Old full-range
   signatures stay (thin wrappers) so tests and `--no-capture` paths are untouched.
3. **`src/core/verify.cpp` + `include/strata/core/verify.hpp`** — THE production decode path for native
   packs: `Verifier` records the window per side (two `exec_[T]` tables), `run()` launches side 0's half,
   serves its rings/pool/flags, enqueues the boundary `R` copy for all `T` rows, launches side 1's half,
   serves it; embedding stays side 0, logits/head/sampler/`R_` move to side 1; `commit()` runs per side.
4. **`src/prefill/prefill.cpp` + hpp** — `Prefill` takes a side's `wt`/`ss`/`xcache`/stream and a layer
   range; the chunk loop runs `[begin, end)`; at the boundary the chunk's residual crosses once per chunk;
   cuBLAS handle, MMQ and the staging ring are per side (the ring streams only its side's experts over its
   own PCIe link — this is where TTFT improves).
5. **`src/core/mtp.cpp` + hpp** — the drafter runs after layer 47: all its allocations and captured graphs
   on side 1; `bind()` takes side 1's table and the side-1 verifier's `final_R_all()`.
6. **`src/program/generate.cpp`** — a `GpuSide sides[2]` scaffold (`ordinal`, `stream`, `wt`, arenas,
   `ss`, `xcache`, hit buffers, `d_res`, `NativeDense`, `logits`/`d_emb` placement); every today-singleton
   allocation becomes a loop over sides under a `DeviceGuard(ordinal)`; `ConvCheckpoint` saves/restores
   per side; `--split-layers` parsing and validation; startup telemetry prints both cards.
7. **`serve/server.py` + `setup.py`** — config `"gpus": [a, b]` (existing `"gpu"` still honored);
   `child_env` sets `CUDA_VISIBLE_DEVICES=a,b`; setup writes both indices; telemetry watches both NVML
   indices; START-HERE/setup.sh pass `--split-layers 24` when two cards are configured.
8. **`tools/`** — `bench_dual.py`: same prompt, greedy, `--split-layers 0` vs `24`, asserts identical
   tokens, reports tok/s + TTFT + per-card VRAM (this is the acceptance gate).

## Order of implementation (each step leaves the tree consistent)

P1 weights + session ranges (items 1-2) — pure additions, nothing call them yet.
P2 generate.cpp scaffold + expert caches + checkpoints (6, parts of 7).
P3 verify split (3) — decode works single-prompt.
P4 prefill split (4) — TTFT path.
P5 MTP split (5) — spec decode (required for native packs).
P6 server/setup/docs/bench (7-8).

## Invariants the review should check

* With no `--split-layers`, the sequence of CUDA calls is IDENTICAL to before (the sides loop runs once,
  ordinal 0, one skip-less load, one session carve `[0,48)`).
* The boundary copy is stream-ordered, never a host sync on the token path.
* `h_seq` numbering stays global; the host never resets it mid-token.
* A layer's `parts` is always that layer's side's `parts_dev`.
* Greedy tokens from `--split-layers 24` match `--split-layers 0` bit-for-bit on the same pack (modulo the
  documented GPU-vs-CPU expert rounding, which already differs run-to-run with the cache on).

## Verification (on the 2x3060 box)

1. Build: `python3 setup.py` (or the existing flow) — must compile clean on sm_86.
2. Regression: today's exact command, no flag — tok/s and tokens within noise of the installed engine.
3. Correctness: `tools/bench_dual.py` — greedy, fixed seed, split vs no-split token streams identical.
4. Speed: 128K-context conversation, IQ3_XXS: expect ~38-45 tok/s (from 31) and a shorter TTFT; per-card
   VRAM via nvidia-smi should show both cards near-full expert caches.
