#!/usr/bin/env python3
"""tools/bench_dual.py - the dual-GPU acceptance gate (docs/DUAL-GPU.md).

Runs the SAME engine command twice - once plain, once with `--split-layers N` injected - on the same
greedy prompt, then:

  1. asserts the two token streams are IDENTICAL (the split must not change the model's answer);
  2. reports tok/s and TTFT for both arms plus per-card VRAM, from the engine's own stderr lines.

The engine command is whatever you actually run, MINUS the flag this script injects.  The prompt is
piped to the engine on stdin in its `--serve` protocol (`GEN <max_new> <id,...>`, tokens back as
`T <id>`, `DONE ...`) because that output is line-structured and stable; if your command is not a
serve command, pass --raw and the script falls back to timing only (no token comparison).

Example (adjust paths to the install):

    python3 tools/bench_dual.py --split 24 \
        --cmd engine/strata generate --serve --pack <pack> --ple-gguf <p> --native <s> \
               --spec 4 --mtp <mtp> --prefill auto --expert-profile data/expert-profile.bin

Exits 0 when the streams match (or --raw), 1 when they differ, 2 on usage/engine failure.
"""
import argparse
import re
import subprocess
import sys
import time


def run_arm(cmd: list[str], prompt: str, timeout_s: int) -> tuple[list[int], float, float, str]:
    """Returns (tokens, ttft_s, total_s, stderr_tail)."""
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, input=prompt.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout_s)
    total = time.perf_counter() - t0
    out = proc.stdout.decode(errors="replace")
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode(errors="replace")[-2000:])
        sys.exit(f"bench_dual: the engine exited {proc.returncode}")
    toks: list[int] = []
    ttft = float("nan")
    t_launch = t0
    for line in out.splitlines():
        if line.startswith("T "):
            if len(toks) == 0:
                ttft = time.perf_counter() - t_launch  # includes prefill; honest TTFT
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            t_launch = time.perf_counter()  # not used; DONE closes the request
    return toks, ttft, total, proc.stderr.decode(errors="replace")[-4000:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", nargs=argparse.REMAINDER, required=True,
                    help="the engine command, without --split-layers")
    ap.add_argument("--split", type=int, default=24, help="the N to inject (default 24)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=str,
                    default="64,9707,11,1879,30,278,1996,1520,499,307,596,1029,11,314,1109,1632,13",
                    help="comma-separated token ids of the greedy prompt (a plain factual question)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--raw", action="store_true", help="do not compare token streams")
    args = ap.parse_args()
    if not args.cmd:
        ap.error("--cmd needs the engine command after it")

    prompt = f"GEN {args.max_new} {args.prompt_tokens}\nQUIT\n"
    arms = {"single": list(args.cmd), "split": list(args.cmd) + ["--split-layers", str(args.split)]}
    results = {}
    for name, cmd in arms.items():
        print(f"[bench_dual] running the {name} arm: {' '.join(cmd[:6])} ...", file=sys.stderr)
        toks, ttft, total, tail = run_arm(cmd, prompt, args.timeout)
        results[name] = (toks, ttft, total, tail)
        n = len(toks)
        print(f"[bench_dual] {name}: {n} tokens, TTFT {ttft:.1f}s, {n / total if total else 0:.1f} tok/s wall")
        for line in tail.splitlines():
            if re.search(r"expert cache|dual-GPU|split", line):
                print(f"    | {line}")

    a, b = results["single"][0], results["split"][0]
    if args.raw:
        return 0
    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"[bench_dual] MISMATCH at token {i}: single {x} vs split {y}")
                break
        print(f"[bench_dual] FAIL: streams differ (lengths {len(a)} vs {len(b)}).")
        print("              Note: with the expert cache ON, GPU-vs-CPU rounding can flip a near-tie")
        print("              (bench/results/2026-09-27-cache-parity); rerun both arms with --expert-cache 0")
        print("              for a bit-exact comparison.")
        return 1
    print(f"[bench_dual] PASS: {len(a)} identical tokens across both arms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
