#!/usr/bin/env python3
"""generate_qa.py — Bulk synthetic Q&A generation for MITOS-QA via a running vLLM server.

Spec: AGENTS.md (§4 flow, §5 output schema, §6 performance, §7 robustness, §8 interface).

For every input JSONL row:
  * one chat request per round with ``n`` = still-needed samples (initial round: n=5)
    → 1 prefill → n parallel samples (§4.2);
  * every sample is parsed into «Ερώτηση: …» / «Απάντηση: …», deduplicated
    (normalized text + difflib ratio ≥ threshold) and written as one JSONL row;
  * duplicates/invalid samples are topped up with extra requests (new seed) up to
    ``--max-samples-per-doc`` total samples; rows that never reach ``--samples``
    unique pairs are logged to failures.jsonl and the flow continues (§7);
  * the «Πηγή: [source](source_url)» line is appended by this client — never by
    the model (any Πηγή line the model produced is stripped first, §4.6);
  * output is append-only, flushed per batch; a re-run resumes automatically
    (docs with ≥ ``--samples`` valid pairs are skipped, the rest are topped up).

Run from the login node with the venv that contains aiohttp:

    VLLM/vllm_cpu_env/bin/python generate_qa.py \
        --input  data/merged_unfiltered_dataset.jsonl \
        --output data/synthetic_qa.jsonl \
        --node m42 --concurrency 192 --samples 5 --max-tokens 1024
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import getpass
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Any

import aiohttp

# --------------------------------------------------------------------------- #
# Σταθερές
# --------------------------------------------------------------------------- #
JOB_NAME = "vllm_cpu"  # όνομα του SLURM job που τρέχει τον vLLM server
DEFAULT_PORT = 8000
MIN_QUESTION_CHARS = 10  # ελάχιστο μήκος έγκυρης ερώτησης (§7)
MIN_ANSWER_CHARS = 40  # ελάχιστο μήκος έγκυρης απάντησης (§7)

# --------------------------------------------------------------------------- #
# Prompt (§4.1)
# --------------------------------------------------------------------------- #
PROMPT_HEADER = """Από τα παρακάτω στοιχεία (τίτλο, περιγραφή, προϋποθέσεις) δημιούργησε μια σαφή, ρεαλιστική ερώτηση και μια απάντηση για αυτήν. Να βασιστείς αποκλειστικά σε αυτά, χωρίς εξωτερικές γνώσεις· απάντησε στα ελληνικά.
Η απάντησή σου να έχει αποκλειστικά τη μορφή «Ερώτηση: …» / «Απάντηση: …» — σύντομη, αυτοτελής, δομημένη σε Markdown (σαν καταχώριση FAQ). Μη συμπεριλάβεις τίποτε άλλο (ούτε σχόλια, ούτε «Πηγή» ή συνδέσμους)."""


def _field(value: Any) -> str:
    """String του πεδίου, trimmed ('' για None)."""
    return "" if value is None else str(value).strip()


def _is_missing(value: Any) -> bool:
    """Missing = None / κενό / 'null' / 'none' / 'nan'."""
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"null", "none", "nan"}


def _strip_leading_label(text: str, label: str) -> str:
    """Πολλά rows έχουν το label ενσωματωμένο (π.χ. «Περιγραφή: …») — αφαιρείται για να μη διπλασιαστεί."""
    m = re.match(rf"^\s*{re.escape(label)}\s*[:：]\s*", text, flags=re.IGNORECASE)
    return text[m.end():].strip() if m else text


def build_prompt(row: dict[str, Any]) -> str:
    """Χτίζει το user prompt (§4.1): οδηγία + τίτλος + περιγραφή + προϋποθέσεις."""
    title = _field(row.get("title_el"))
    desc = _strip_leading_label(_field(row.get("simple_description")), "Περιγραφή")
    parts = [PROMPT_HEADER, f"Τίτλος: {title}", f"Περιγραφή: {desc}"]
    reqs = "" if _is_missing(row.get("requirements")) else _field(row.get("requirements"))
    if reqs:
        parts += ["", "Προϋποθέσεις (requirements):", reqs]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Parsing απάντησης μοντέλου (§4 βήμα 3)
# --------------------------------------------------------------------------- #
_QM_LINE = re.compile(r"(?im)^[ \t>*_#-]*Ερώτηση\s*[:：]\s*")
_AM_LINE = re.compile(r"(?im)^[ \t>*_#-]*Απάντηση\s*[:：]\s*")
_QM_ANY = re.compile(r"Ερώτηση\s*[:：]\s*")
_AM_ANY = re.compile(r"Απάντηση\s*[:：]\s*")
_SOURCE_LINE_RE = re.compile(r"(?im)^[ \t>*#-]*(?:\*\*|__)?\s*Πηγή\s*(?:\*\*|__)?\s*[:：].*$")
# Κλείσιμο bold/italic ακριβώς μετά το marker (π.χ. «**Απάντηση:**\n…») — μόνο όταν
# ακολουθεί κενό/newline/EOS, ώστε να μη σπάσει έμφαση τύπου «**Σημαντικό:**…».
_LEAD_BOLD_RE = re.compile(r"^(?:\*\*|__)(?=\s|$)[ \t]*\n?")
_FRAGMENT_STRIP = "«»\"'*_#` \t\r\n"


def _clean_question(s: str) -> str:
    return s.strip(_FRAGMENT_STRIP)


def _cut_second_pair(answer: str) -> str:
    """Αν το μοντέλο έβγαλε και δεύτερο «Ερώτηση: …» στην απάντηση, κόβεται (κρατάμε το πρώτο ζεύγος)."""
    m = _QM_LINE.search(answer)
    return answer[: m.start()].rstrip() if m else answer


def _fix_unbalanced_fence(answer: str) -> str:
    """Αφαίρεσε ένα περιττό μεμονωμένο ``` (π.χ. wrapper του μοντέλου γύρω από όλη την απάντηση)."""
    lines = answer.split("\n")
    fences = [i for i, ln in enumerate(lines) if ln.strip().startswith("```")]
    if len(fences) % 2 == 1:
        lines.pop(fences[-1])
        return "\n".join(lines).strip()
    return answer


def strip_source_lines(text: str) -> str:
    """Αφαίρεσε τυχόν «Πηγή: …» γραμμές — τη σωστή Πηγή την προσθέτει ο client (§4 βήμα 6)."""
    out = _SOURCE_LINE_RE.sub("", text)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def parse_choice(content: str) -> tuple[str, str] | None:
    """Εξάγει («Ερώτηση», «Απάντηση») από ένα message.content· None αν δεν ταιριάζει η μορφή."""
    if not content:
        return None
    m_q = _QM_LINE.search(content) or _QM_ANY.search(content)
    if m_q is None:
        return None
    rest = content[m_q.end():]
    m_a = _AM_LINE.search(rest) or _AM_ANY.search(rest)
    if m_a is None:
        return None
    question = _clean_question(strip_source_lines(rest[: m_a.start()]))
    answer = _cut_second_pair(rest[m_a.end():]).strip()
    answer = _LEAD_BOLD_RE.sub("", answer).strip()
    answer = _fix_unbalanced_fence(strip_source_lines(answer))
    if not question or not answer:
        return None
    return question, answer


def is_valid_pair(question: str, answer: str) -> bool:
    """Έγκυρο ζεύγος: μη κενό και πάνω από ένα ελάχιστο μήκος (§7)."""
    return len(question) >= MIN_QUESTION_CHARS and len(answer) >= MIN_ANSWER_CHARS


# --------------------------------------------------------------------------- #
# Dedup (§4 βήμα 4) & seeds (§7)
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(s: str) -> str:
    """Κανονικοποίηση: πεζά, χωρίς τόνους/στίξη/διπλά κενά."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("ς", "σ")
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_duplicate(norm: str, existing: list[str], threshold: float) -> bool:
    """Ίδιο μετά την κανονικοποίηση ή SequenceMatcher ratio ≥ threshold."""
    for other in existing:
        if norm == other:
            return True
        if difflib.SequenceMatcher(None, norm, other).ratio() >= threshold:
            return True
    return False


def stable_seed(doc_id: str, sample_index: int) -> int:
    """seed = hash(doc_id, i): crc32 ώστε να είναι σταθερό μεταξύ runs (το builtin hash() είναι randomized)."""
    return zlib.crc32(f"{doc_id}#{sample_index}".encode("utf-8")) & 0x7FFFFFFF


def append_source(answer: str, source: str, source_url: str) -> str:
    """Προσθέτει «Πηγή: [source](source_url)» στο τέλος της απάντησης (μόνο ο client, §4 βήμα 6)."""
    if _is_missing(source_url):
        return answer
    label = source if not _is_missing(source) else source_url
    return f"{answer.rstrip()}\n\nΠηγή: [{label}]({source_url})"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def format_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# --------------------------------------------------------------------------- #
# Σφάλματα & writers
# --------------------------------------------------------------------------- #
class RetryableError(Exception):
    """Προσωρινό σφάλμα (timeout, 5xx, 429, άκυρη JSON) — retry με backoff."""


class PermanentError(Exception):
    """Μόνιμο σφάλμα (4xx) — δεν έχει νόημα retry."""


class AttemptsExhausted(Exception):
    """Εξαντλήθηκαν οι προσπάθειες για ένα request."""


class JsonlWriter:
    """JSONL writer με flush ανά batch (append-only, ασφαλές σε διακοπή — §5/§7)."""

    def __init__(self, path: Path, mode: str = "a") -> None:
        self.path = path
        self._mode = mode
        self._fh = None
        self._lock = asyncio.Lock()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(self._mode, encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    async def write_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        async with self._lock:
            for row in rows:
                self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()


class Stats:
    """Ζωντανή μέτρηση: in-flight, docs/min, tok/s από το usage κάθε response (§6.3)."""

    def __init__(self, pending: int) -> None:
        self.pending = pending
        self.processed = 0
        self.failed = 0
        self.skipped = 0
        self.in_flight = 0
        self.pairs_written = 0
        self.requests_ok = 0
        self.retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.t0 = time.monotonic()
        self._last_t = self.t0
        self._last_processed = 0
        self._last_tokens = 0

    def add_usage(self, usage: dict[str, Any]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def progress_line(self) -> str:
        now = time.monotonic()
        dt = max(now - self._last_t, 1e-6)
        rate_tokens = (self.completion_tokens - self._last_tokens) / dt
        rate_docs = (self.processed - self._last_processed) * 60.0 / dt
        self._last_t, self._last_processed, self._last_tokens = (
            now,
            self.processed,
            self.completion_tokens,
        )
        elapsed = max(now - self.t0, 1e-6)
        return (
            f"[progress] {self.processed}/{self.pending} docs (failed {self.failed})"
            f" | in-flight {self.in_flight} | pairs {self.pairs_written}"
            f" | docs/min {rate_docs:.1f} (avg {self.processed * 60.0 / elapsed:.1f})"
            f" | out {self.completion_tokens / 1000:.1f}k tok"
            f" | tok/s {rate_tokens:.1f} (avg {self.completion_tokens / elapsed:.1f})"
        )


# --------------------------------------------------------------------------- #
# Runner: request → parse → dedup → top-up → εγγραφή
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(
        self,
        args: argparse.Namespace,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        writer: JsonlWriter,
        failures: JsonlWriter,
        resume: dict[str, dict[str, Any]],
        stats: Stats,
    ) -> None:
        self.args = args
        self.session = session
        self.base_url = base_url
        self.model = model
        self.writer = writer
        self.failures = failures
        self.resume = resume
        self.stats = stats
        self.queue: asyncio.Queue = asyncio.Queue()
        self.abort_reason: str | None = None
        self._request_failures = 0

    async def fail(self, doc_id: str, reason: str, kept: int, critical: bool = False) -> None:
        """Καταγραφή αποτυχίας στο failures.jsonl + ροή συνεχίζει (§7)."""
        self.stats.failed += 1
        entry = {
            "doc_id": doc_id,
            "reason": reason,
            "kept": kept,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        await self.failures.write_many([entry])
        log(f"[fail] {doc_id}: {reason}")
        if critical:
            self._request_failures += 1
            if self.stats.requests_ok == 0 and self._request_failures >= self.args.abort_after:
                # Συστηματικό σφάλμα (π.χ. λάθος model id / WORKDIR) — μη γεμίσεις το failures.jsonl.
                self.abort_reason = (
                    f"{self._request_failures} request failures before any success — "
                    f"check --model / server WORKDIR ({reason})"
                )
                log(f"[abort] {self.abort_reason}")

    async def request_qa(self, doc_id: str, prompt: str, n: int, seed: int) -> dict[str, Any]:
        """POST /v1/chat/completions με n samples + retries με backoff (§4 βήμα 2, §6.3)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_tokens": self.args.max_tokens,
            "seed": seed,
            "stream": False,
        }
        url = f"{self.base_url}/v1/chat/completions"
        last_err: Exception | None = None
        for attempt in range(1, self.args.max_attempts + 1):
            try:
                async with self.session.post(url, json=payload) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        try:
                            data = json.loads(body)
                        except json.JSONDecodeError as e:
                            raise RetryableError(f"invalid JSON: {e}") from e
                        if not data.get("choices"):
                            raise RetryableError("response has no choices")
                        return data
                    if resp.status == 429 or resp.status >= 500:
                        raise RetryableError(f"HTTP {resp.status}: {body[:160]}")
                    raise PermanentError(f"HTTP {resp.status}: {body[:160]}")
            except PermanentError:
                raise
            except (RetryableError, asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_err = e
                if attempt >= self.args.max_attempts:
                    break
                self.stats.retries += 1
                delay = min(60.0, self.args.retry_base * 2 ** (attempt - 1)) + random.uniform(
                    0.0, self.args.retry_base
                )
                log(
                    f"[retry {attempt}/{self.args.max_attempts}] {doc_id}: "
                    f"{type(e).__name__}: {e} — backoff {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        raise AttemptsExhausted(f"{type(last_err).__name__}: {last_err}")

    async def process_doc(self, row: dict[str, Any]) -> None:
        args = self.args
        doc_id = str(row.get("doc_id", "")).strip()
        st = self.resume.get(doc_id)
        kept = int(st["count"]) if st else 0
        norms: list[str] = list(st["norms"]) if st else []
        if kept >= args.samples:
            return  # δεν θα συμβεί (έχει φιλτραριστεί) — ασφάλεια

        prompt = build_prompt(row)
        source = _field(row.get("source"))
        source_url = _field(row.get("source_url"))
        produced = kept  # πλήθος δειγμάτων που έχουν ζητηθεί (approx όταν γίνεται resume)

        while kept < args.samples:
            if produced >= args.max_samples_per_doc:
                await self.fail(
                    doc_id,
                    f"not enough unique pairs after {produced} samples "
                    f"(kept {kept}/{args.samples})",
                    kept,
                )
                return

            need = args.samples - kept
            seed = stable_seed(doc_id, produced)  # νέο seed ανά round (§4 βήμα 5)
            try:
                data = await self.request_qa(doc_id, prompt, need, seed)
            except AttemptsExhausted as e:
                await self.fail(
                    doc_id, f"request failed after {args.max_attempts} attempts: {e}", kept, critical=True
                )
                return
            except PermanentError as e:
                await self.fail(doc_id, f"request rejected: {e}", kept, critical=True)
                return

            self.stats.requests_ok += 1
            self.stats.add_usage(data.get("usage") or {})
            choices = data.get("choices") or []
            produced += len(choices)

            new_rows: list[dict[str, Any]] = []
            for choice in choices:
                content = ((choice.get("message") or {}).get("content")) or ""
                parsed = parse_choice(content)
                if parsed is None:
                    continue
                question, answer = parsed
                if not is_valid_pair(question, answer):
                    continue
                norm = normalize_text(question)
                if is_duplicate(norm, norms, args.dedup_threshold):
                    continue
                norms.append(norm)
                out_row = dict(row)  # τα πεδία της γραμμής περνούν αυτούσια (§3)
                out_row["question"] = question
                out_row["answer"] = append_source(answer, source, source_url)
                new_rows.append(out_row)
                kept += 1

            if new_rows:
                await self.writer.write_many(new_rows)  # append + flush αμέσως (§4 βήμα 6)
                self.stats.pairs_written += len(new_rows)

            if args.verbose:
                log(f"[doc] {doc_id}: +{len(new_rows)} → {kept}/{args.samples} (produced {produced})")

    async def worker(self) -> None:
        while self.abort_reason is None:
            try:
                row = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self.stats.in_flight += 1
            try:
                await self.process_doc(row)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # unexpected — ποτέ να μη σκοτώνει τη ροή (§6.3)
                await self.fail(
                    str(row.get("doc_id", "")), f"unexpected error: {type(e).__name__}: {e}", -1
                )
            finally:
                self.stats.in_flight -= 1
                self.stats.processed += 1


# --------------------------------------------------------------------------- #
# I/O & εντοπισμός server (§2)
# --------------------------------------------------------------------------- #
def detect_node() -> str:
    """Εντοπισμός compute node από το RUNNING vllm_cpu job (§2)."""
    user = os.environ.get("USER") or getpass.getuser()
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-n", JOB_NAME, "-h", "-t", "RUNNING", "-o", "%N"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as e:
        raise SystemExit(
            "ERROR: `squeue` not found — run on the ARIS login node, or pass --node/--base-url"
        ) from e
    node = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    if not node:
        raise SystemExit(
            f"ERROR: no RUNNING `{JOB_NAME}` job for user {user} — "
            "start the vLLM server first, or pass --node/--base-url"
        )
    return node


def resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    node = args.node
    if not node:
        node = detect_node()
        log(f"[info] node auto-detected from RUNNING {JOB_NAME} job: {node}")
    return f"http://{node}:{DEFAULT_PORT}"


def load_input(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"[warn] {path}:{i}: malformed JSON, skipped ({e})")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_resume(path: Path, samples: int) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Resume (§7): διάβασε το υπάρχον output → (state ανά doc_id, pairs, docs_complete)."""
    state: dict[str, dict[str, Any]] = {}
    pairs = 0
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log(f"[warn] {path}:{i}: malformed line in existing output, ignored")
                continue
            doc_id = str(row.get("doc_id", "")).strip()
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not doc_id or not question or not answer:
                continue
            st = state.setdefault(doc_id, {"count": 0, "norms": []})
            st["count"] += 1
            st["norms"].append(normalize_text(question))
            pairs += 1
    complete = sum(1 for st in state.values() if st["count"] >= samples)
    return state, pairs, complete


async def wait_for_server(session: aiohttp.ClientSession, base_url: str, wait_s: float) -> list[str]:
    """Έλεγχος ετοιμότητας (§2): GET /v1/models μέχρι να απαντήσει ο server."""
    deadline = time.monotonic() + wait_s
    announced = False
    while True:
        try:
            async with session.get(
                f"{base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [
                        m["id"]
                        for m in (data.get("data") or [])
                        if isinstance(m, dict) and m.get("id")
                    ]
                    if models:
                        return models
                    log("[wait] /v1/models OK but empty model list — retrying...")
                else:
                    log(f"[wait] /v1/models -> HTTP {resp.status} — retrying...")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if not announced:
                log(
                    f"[wait] vLLM server not ready yet at {base_url} — "
                    "if this is the first start, AOT compile can take several minutes"
                )
                announced = True
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"ERROR: vLLM server did not answer /v1/models at {base_url} within {wait_s:.0f}s"
            )
        await asyncio.sleep(10.0)


async def progress_loop(stats: Stats, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        log(stats.progress_line())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
async def run(args: argparse.Namespace) -> int:
    base_url = resolve_base_url(args)
    log(f"[info] base URL: {base_url}")

    input_rows = load_input(Path(args.input))
    if not input_rows:
        raise SystemExit(f"ERROR: no usable rows in {args.input}")
    log(f"[info] input: {args.input} — {len(input_rows)} rows")

    resume: dict[str, dict[str, Any]] = {}
    complete_docs = 0
    out_path = Path(args.output)
    if out_path.exists():
        resume, resumed_pairs, complete_docs = load_resume(out_path, args.samples)
        log(
            f"[info] resume: {resumed_pairs} existing pairs for {len(resume)} docs"
            f" ({complete_docs} complete, skipped)"
        )
    else:
        log(f"[info] output: {args.output} (new file)")

    pending = [
        r
        for r in input_rows
        if resume.get(str(r.get("doc_id", "")).strip(), {}).get("count", 0) < args.samples
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        log(f"[info] nothing to do — every input row already has >= {args.samples} pairs")
        return 0
    log(
        f"[info] pending docs: {len(pending)} | concurrency: {args.concurrency}"
        f" | samples: {args.samples} | max_tokens: {args.max_tokens}"
        f" | temp: {args.temperature} | top_p: {args.top_p}"
    )
    log(f"[info] failures -> {args.failures}")

    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=30.0)
    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        models = await wait_for_server(session, base_url, args.server_wait)
        if args.model:
            model = args.model
            if model not in models:
                log(
                    f"[warn] --model '{model}' is not among {models} — "
                    "θα χρησιμοποιηθεί όπως δόθηκε (404 = λάθος id / WORKDIR)"
                )
        else:
            model = models[0]
            log(f"[info] model (from GET /v1/models): {model}")

        writer = JsonlWriter(out_path, mode="a")
        writer.open()
        failures = JsonlWriter(Path(args.failures), mode="w")
        failures.open()

        stats = Stats(pending=len(pending))
        stats.skipped = complete_docs
        runner = Runner(args, session, base_url, model, writer, failures, resume, stats)
        for row in pending:
            runner.queue.put_nowait(row)

        workers = [
            asyncio.create_task(runner.worker(), name=f"worker-{i}")
            for i in range(min(args.concurrency, len(pending)))
        ]
        progress_task = asyncio.create_task(progress_loop(stats, args.log_interval))

        log(f"[info] starting {len(workers)} workers...")
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
            writer.close()
            failures.close()

    elapsed = time.monotonic() - stats.t0
    log(
        f"[summary] docs: {stats.processed}/{stats.pending} processed"
        f" | ok {stats.processed - stats.failed} | failed {stats.failed}"
        f" | skipped (already complete) {stats.skipped}"
    )
    log(
        f"[summary] pairs written: {stats.pairs_written}"
        f" | requests: {stats.requests_ok} (retries {stats.retries})"
        f" | tokens out/prompt: {stats.completion_tokens}/{stats.prompt_tokens}"
    )
    log(
        f"[summary] elapsed {format_hms(elapsed)}"
        f" | avg {stats.completion_tokens / max(elapsed, 1e-6):.1f} out tok/s"
    )
    if stats.failed:
        log(
            f"[summary] {stats.failed} failures in {args.failures} — "
            "re-run the same command to retry them (completed docs are skipped)"
        )
    if runner.abort_reason:
        log(f"[abort] {runner.abort_reason}")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk synthetic Q&A generation for MITOS-QA via a running vLLM server "
        "(AGENTS.md §4–§7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("input/output")
    g.add_argument("--input", default="data/merged_unfiltered_dataset.jsonl", help="input JSONL")
    g.add_argument(
        "--output", default="data/synthetic_qa.jsonl", help="output JSONL (append-only, resume)"
    )
    g.add_argument(
        "--failures", default=None, help="failure log JSONL (default: failures.jsonl next to --output)"
    )

    g = p.add_argument_group("server")
    g.add_argument(
        "--node",
        default=None,
        help=f"compute node of the vLLM server (default: auto-detect from RUNNING {JOB_NAME} job)",
    )
    g.add_argument("--base-url", default=None, help=f"e.g. http://m42:{DEFAULT_PORT} (overrides --node)")
    g.add_argument("--model", default=None, help="model id (default: first id from GET /v1/models)")
    g.add_argument("--server-wait", type=float, default=600.0, help="seconds to wait for GET /v1/models")

    g = p.add_argument_group("generation")
    g.add_argument("--samples", type=int, default=5, help="unique QA pairs required per input row")
    g.add_argument(
        "--max-samples-per-doc", type=int, default=10, help="give up on a row after this many samples"
    )
    g.add_argument("--max-tokens", type=int, default=1024, help="max_tokens per generated sample")
    g.add_argument("--temperature", type=float, default=0.9, help="0.8–1.0; never <= 0 with n>1")
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument(
        "--dedup-threshold", type=float, default=0.8, help="SequenceMatcher ratio for near-duplicates"
    )

    g = p.add_argument_group("client behaviour")
    g.add_argument(
        "--concurrency",
        type=int,
        default=192,
        help="in-flight requests: 128–256 on a TP=8 server, 32–128 on a single-rank server",
    )
    g.add_argument("--max-attempts", type=int, default=3, help="attempts per HTTP request (incl. first)")
    g.add_argument("--retry-base", type=float, default=3.0, help="base seconds for exponential backoff")
    g.add_argument(
        "--request-timeout",
        type=float,
        default=7200.0,
        help="per-request total timeout (s) — keep high: requests wait in the server queue",
    )
    g.add_argument(
        "--abort-after",
        type=int,
        default=5,
        help="abort the run if this many request-level failures occur before any success",
    )
    g.add_argument("--limit", type=int, default=None, help="process at most N pending rows (test run)")
    g.add_argument("--log-interval", type=float, default=30.0, help="progress log interval (s)")
    g.add_argument("-v", "--verbose", action="store_true", help="log per-row results")

    args = p.parse_args(argv)

    if args.samples < 1:
        p.error("--samples must be >= 1")
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    if args.temperature <= 0:
        p.error(
            "--temperature must be > 0 (with n>1, temperature=0 would yield identical samples — "
            "see AGENTS.md §4)"
        )
    if args.max_samples_per_doc < args.samples:
        p.error("--max-samples-per-doc must be >= --samples")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1")
    if args.failures is None:
        args.failures = str(Path(args.output).with_name("failures.jsonl"))
    return args


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(run(args))
    except KeyboardInterrupt:
        log("[interrupted] output is flushed per record — re-run the same command to resume")
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
