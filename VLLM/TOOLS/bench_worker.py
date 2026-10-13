"""Benchmark worker for TOOLS/bench_dtype.sh (one dtype per invocation).

Must be a real file (not a `python -` heredoc): vLLM's engine core runs as a
multiprocessing "spawn" child, which re-opens the parent's __main__ file; a
stdin script would crash with `FileNotFoundError ... '<stdin>'`.

The __main__ guard keeps the child from re-running the benchmark.
"""

import os
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    dtype = os.environ["DTYPE"]
    n = int(os.environ["N_TOKENS"])
    model = os.environ["MODEL"]
    max_len = int(os.environ["MAX_LEN"]) if os.environ.get("MAX_LEN") else None

    t0 = time.time()
    llm = LLM(model=model, dtype=dtype, max_model_len=max_len)
    t1 = time.time()
    print(f"[{dtype}] engine init (weights+compile+warmup): {t1 - t0:.0f}s", flush=True)

    sp = SamplingParams(temperature=0, max_tokens=n)
    out = llm.generate(os.environ["PROMPT"], sp)  # warm
    t2 = time.time()
    print(f"[{dtype}] warm generate: {t2 - t1:.1f}s", flush=True)

    t3 = time.time()
    out = llm.generate(os.environ["PROMPT"], sp)
    t4 = time.time()
    ntok = len(out[0].outputs[0].token_ids)
    print(
        f"[{dtype}] measured generate: {ntok} tok in {t4 - t3:.1f}s "
        f"=> {ntok / max(t4 - t3, 1e-9):.2f} tok/s (incl. prefill)",
        flush=True,
    )


if __name__ == "__main__":
    main()
