"""Throughput benchmark: one engine, increasing batch sizes (offline batching).

Prompts share a common prefix (like a QA system prompt) and differ only in a
suffix, so prefix caching behaves realistically. Aggregated tok/s shows how
many questions the server can answer per second - the metric that matters for
batch-processing a QA dataset.
"""

import os
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    model = os.environ["MODEL"]
    n = int(os.environ["N_TOKENS"])
    base = os.environ["PROMPT"]

    t0 = time.time()
    llm = LLM(model=model, dtype=os.environ.get("DTYPE", "bfloat16"))
    print(f"engine init: {time.time() - t0:.0f}s", flush=True)

    sp = SamplingParams(temperature=0, max_tokens=n)
    llm.generate([f"{base} (#0)"], sp)  # warm

    for b in (1, 4, 16, 32):
        prompts = [f"{base} (#{i})" for i in range(b)]
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(
            f"[batch={b:2d}] {ntok} tok in {dt:6.1f}s -> aggregate {ntok / dt:6.1f} tok/s"
            f"  (per-stream {ntok / dt / b:.2f} tok/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
