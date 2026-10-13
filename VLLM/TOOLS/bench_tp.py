"""Tensor-parallel benchmark across NUMA nodes (CPU backend)."""

import os
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    model = os.environ["MODEL"]
    n = int(os.environ["N_TOKENS"])
    tp = int(os.environ["TP"])
    prompt = os.environ["PROMPT"]

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=tp)
    print(f"[tp={tp}] engine init: {time.time() - t0:.0f}s", flush=True)

    sp = SamplingParams(temperature=0, max_tokens=n)
    llm.generate([prompt], sp)  # warm

    t0 = time.time()
    outs = llm.generate([prompt], sp)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[tp={tp} single] {ntok} tok in {dt:.1f}s -> {ntok / dt:.2f} tok/s", flush=True)

    b = 8
    prompts = [f"{prompt} (#{i})" for i in range(b)]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"[tp={tp} batch={b}] {ntok} tok in {dt:.1f}s -> aggregate {ntok / dt:.1f} tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
