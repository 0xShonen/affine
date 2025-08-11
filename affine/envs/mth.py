import re
import time
import random
import textwrap
from fractions import Fraction
from typing import Optional, Tuple

import affine as af
from affine.utils.sandbox import SandboxManager

# Regexes
ANSWER_RE = re.compile(r"Answer\s*:?\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
PROOF_RE = re.compile(r"```lean\n(.*?)```", re.DOTALL)


def _extract_numeric(text: str) -> Optional[Fraction]:
    if not text:
        return None
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    return Fraction(m.group(1)) if m else None


class MTH(af.BaseEnv):
    """
    GSM8K math problems with a Lean 4 proof requirement.
    Score = 1.0 only if (i) numeric answer matches and
    (ii) the Lean proof compiles inside a Docker sandbox with `lean --make`.
    """

    __version__: str = "0.2.1"

    def __init__(self, split: str = "train"):
        super().__init__(split=split)
        # Buffered HF dataset (GSM8K: config "main")
        total = 7473 if split == "train" else 1319
        self._data = af.utils.BufferedDataset(
            dataset_name="openai/gsm8k",
            total_size=total,
            buffer_size=5,
            max_batch=5,
            split=split,
            config="main",
        )
        self._mgr = SandboxManager(
            image="leanprovercommunity/lean4:latest",
            workdir="/work",
            pull=True,
            tmpfs={"/work": "rw,size=512m", "/tmp": "rw,size=256m"},
            network_disabled=True,
            read_only_root=False,
            mem_limit="1g",
            cpus=1.0,
            label_ns="rl-lean-sandbox",
        )

    async def generate(self) -> af.Challenge:
        # Keep sampling until we hit a numeric-answer example
        while True:
            ex = await self._data.get()
            question = (ex.get("question") or "").strip()
            gold = _extract_numeric(ex.get("answer") or "")
            if not question or gold is None:
                af.logger.trace("MTH.generate: skipping sample; has_question=%s has_numeric=%s", bool(question), gold is not None)
                continue

            prompt = textwrap.dedent(
                f"""
                **Grade-School Math Problem**

                {question}

                ### Instructions
                1. Solve the problem and give the final numeric answer.
                2. Prove your solution in Lean 4.
                   Return your proof in a single fenced block with language `lean`:

                ```lean
                -- your Lean 4 proof here, e.g.:

                theorem solution : True := by
                  trivial
                ```

                3. After the proof block, include a line:
                   `Answer: <number>`

                Exactly one `lean`-fenced block and exactly one `Answer:` line, please.
                
                Note: the answer is {str(gold)}
                """
            ).strip()

            af.logger.trace("MTH.generate: emitting challenge; gold=%s idx=%s", gold, ex.get("idx", None))
            return af.Challenge(
                env=self,
                prompt=prompt,
                extra={
                    "gold": str(gold),
                    "idx": ex.get("idx", None),
                    "timestamp": time.time(),
                },
            )

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        text = response.response or ""
        gold = Fraction(challenge.extra["gold"])  # exact compare
        af.logger.trace("MTH.evaluate: start; gold=%s latency=%.3fs", gold, getattr(response, "latency_seconds", 0.0))

        failure_reasons = []

        # 1) Check numeric answer line
        m = ANSWER_RE.search(text)
        got = Fraction(m.group(1)) if m else None
        answer_ok = (got == gold)
        if not m:
            failure_reasons.append("no_answer_line")
        elif not answer_ok:
            failure_reasons.append("answer_mismatch")
        af.logger.trace("MTH.evaluate: answer_present=%s got=%s answer_ok=%s", bool(m), str(got) if got is not None else None, answer_ok)

        # 2) Compile Lean proof inside sandbox
        proof_ok = False
        compile_info = None
        pm = PROOF_RE.search(text)
        if not pm:
            failure_reasons.append("no_proof_block")
            af.logger.trace("MTH.evaluate: no Lean proof block found")
        else:
            lean_body = pm.group(1)
            payload = textwrap.dedent(lean_body).encode("utf-8")
            af.logger.trace("MTH.evaluate: proof block present; bytes=%d", len(payload))

            start_t = time.monotonic()
            with self._mgr.acquire() as box:
                try:
                    # Ensure workdir exists
                    box.exec(["sh", "-lc", "mkdir -p /work && chmod 1777 /work"])  # ignore exit code
                    # Write file
                    box.put_file_bytes("/work/Solution.lean", payload, mode=0o644)
                    # Attempt to compile; enforce timeout via `timeout` tool
                    code, out = box.exec(["sh", "-lc", "timeout 60 lean --make /work/Solution.lean"])  # type: ignore
                    proof_ok = (code == 0)
                    duration = time.monotonic() - start_t
                    out_text = (out or b"").decode("utf-8", errors="replace")
                    # Truncate potentially large output
                    max_chars = 4000
                    if len(out_text) > max_chars:
                        out_text = out_text[:max_chars] + "…"
                    compile_info = {
                        "exit_code": code,
                        "duration_seconds": round(duration, 4),
                        "output": out_text,
                        "timed_out": (code == 124),
                    }
                    if not proof_ok:
                        failure_reasons.append("lean_compile_failed" if code != 124 else "lean_compile_timeout")
                    af.logger.trace(
                        "MTH.evaluate: lean compile exit_code=%s duration=%.3fs ok=%s",
                        code,
                        duration,
                        proof_ok,
                    )
                except Exception as e:
                    proof_ok = False
                    duration = time.monotonic() - start_t
                    compile_info = {
                        "exception": type(e).__name__,
                        "message": str(e),
                        "duration_seconds": round(duration, 4),
                    }
                    failure_reasons.append("sandbox_exception")
                    af.logger.trace("MTH.evaluate: sandbox exception: %s", e)

        score = 1.0 if (answer_ok and proof_ok) else 0.0
        af.logger.trace("MTH.evaluate: done; score=%.4f reasons=%s", score, failure_reasons)
        extra = {
            "answer_expected": str(gold),
            "answer_received": str(got) if got is not None else None,
            "answer_ok": answer_ok,
            "answer_present": bool(m),
            "proof_ok": proof_ok,
            "proof_present": bool(pm),
            "failure_reasons": failure_reasons,
        }
        if compile_info is not None:
            extra["compile"] = compile_info
        if pm:
            extra["lean_source_len"] = len(pm.group(1))

        return af.Evaluation(
            env=self,
            score=score,
            extra=extra,
        ) 