"""Decisive scaling test: per dtype, batch sizes + real CPU usage sampling.

Reports aggregate tok/s per batch size AND how many CPU cores are actually busy
(sampled from the process tree), to distinguish memory-bound vs (single-)core
compute-bound behavior.
"""

import os
import threading
import time

import psutil


class Sampler:
    """Samples the CPU usage (in 'busy cores') of the whole process tree."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        me = psutil.Process()
        procs = [me, *me.children(recursive=True)]
        for p in procs:
            p.cpu_percent(None)  # prime
        while not self._stop.wait(2.0):
            total = 0.0
            for p in procs:
                try:
                    total += p.cpu_percent(None)
                except psutil.Error:
                    pass
            self._samples.append(total / 100.0)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> tuple[float, float]:
        self._stop.set()
        self._thread.join(timeout=3)
        if not self._samples:
            return 0.0, 0.0
        return sum(self._samples) / len(self._samples), max(self._samples)


def measure(llm, sp, prompts, label) -> None:
    s = Sampler()
    s.start()
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    avg, mx = s.finish()
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"[{label} b={len(prompts):2d}] {ntok:5d} tok in {dt:6.1f}s -> {ntok / dt:6.1f} tok/s"
        f" | busy cores: avg {avg:.1f}, max {mx:.1f}",
        flush=True,
    )


def main() -> None:
    from vllm import LLM, SamplingParams

    dtype = os.environ["DTYPE"]
    model = os.environ["MODEL"]
    tp = int(os.environ.get("TP", "1"))
    max_len = int(os.environ["MAX_LEN"]) if os.environ.get("MAX_LEN") else None
    base = os.environ["PROMPT"]

    t0 = time.time()
    llm = LLM(
        model=model, dtype=dtype, max_model_len=max_len, tensor_parallel_size=tp
    )
    print(f"[{dtype} tp={tp}] engine init: {time.time() - t0:.0f}s", flush=True)

    sp = SamplingParams(temperature=0, max_tokens=64)
    llm.generate([base], sp)  # warm

    batches = [int(x) for x in os.environ.get("BATCHES", "1,8,32").split(",")]
    label = f"{dtype} tp={tp}"
    for b in batches:
        measure(llm, sp, [f"{base} (#{i})" for i in range(b)], label)


if __name__ == "__main__":
    main()
