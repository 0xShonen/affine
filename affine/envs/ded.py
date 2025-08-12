from __future__ import annotations
import re
import ast
import time
import json
import asyncio
import affine as af
from typing import Any, Dict, List, Tuple, Optional
from affine.utils.sandbox import SandboxManager

# -------------------------------- Helpers -------------------------------- #
def _to_str(x) -> str:
    """
    Canonicalise any JSON-serialisable test-case payload to a single
    newline-delimited string suitable for feeding to `stdin`.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    if isinstance(x, list):
        return "\n".join(_to_str(e) for e in x)
    return json.dumps(x, ensure_ascii=False)

def _normalize(text: str) -> str:
    """Trim trailing blank lines and per-line trailing spaces."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())

_FENCE_RE  = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
_HAS_MAIN  = re.compile(r'if\s+__name__\s*==\s*[\'\"]__main__[\'\"]')

# --------------------------------------------------------------------------- #
#                              Affine Env (Docker)                            #
# --------------------------------------------------------------------------- #
class DED(af.BaseEnv):
    __version__: str = "0.1.0-docker"

    def __init__(self):
        super().__init__()
        # Use SandboxManager provided by the Affine runtime
        self._executor = af.utils.ProgramExecutor()
        self._data = af.utils.BufferedDataset(
            dataset_name="satpalsr/rl-python",
            total_size=20_000,
            buffer_size=5,
            max_batch=5,
        )
        self._mgr = SandboxManager(
            image="python:3.11-alpine",
            workdir="/work",
            pull=True,
            tmpfs={"/work": "rw,size=512m", "/tmp": "rw,size=256m"},
            network_disabled=True,
            read_only_root=False,
            mem_limit="1g",
            cpus=1.0,
            label_ns="rl-lean-sandbox",
            max_exec_retries=5,
            healthcheck_cmd="test -x /bin/sh || exit 1",
        )

    # ----------------------------- Env API -------------------------------- #
    async def generate(self) -> af.Challenge:
        af.logger.trace("Generating new coding challenge")
        sample = await self._data.get()
        if sample is None:
            raise RuntimeError("Failed to fetch dataset row")

        extra_hint = (
            "\n\n---\n"
            "⚠️ **Instructions** ⚠️\n"
            "Write a complete **Python 3** program that\n"
            "• reads *all* input from **STDIN** (using `input()` / `sys.stdin`),\n"
            "• writes *only* the required answer(s) to **STDOUT** using `print`,\n"
            "• contains no additional prompts or debug text, and\n"
            "• is returned as a single ```python … ``` fenced block.\n"
        )
        prompt = sample["prompt"].rstrip() + extra_hint
        sample['timestamp'] = time.time()
        return af.Challenge(env=self, prompt=prompt, extra=sample)

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        af.logger.trace("Starting evaluation of the challenge (docker sandbox).")
        raw_reply = response.response
        program = raw_reply or ""
        af.logger.trace(f"Stripped program: {(program or '')[:80]}...")

        # ---------------- Verification info ---------------------------- #
        sample = challenge.extra or {}
        ver_raw = sample.get("verification_info") or sample.get("test_cases")
        af.logger.trace(f"Verification raw: {str(ver_raw)[:80]}...")

        # try JSON first, then Python-literal
        try:
            if isinstance(ver_raw, str):
                try:
                    ver_json = json.loads(ver_raw)
                    af.logger.trace("verification_info parsed via json.loads")
                except json.JSONDecodeError:
                    ver_json = ast.literal_eval(ver_raw)
                    af.logger.trace("verification_info parsed via ast.literal_eval")
            else:
                ver_json = ver_raw
        except Exception as err:
            af.logger.trace(f"Failed to parse verification info: {err}")
            return af.Evaluation(env=self, score=0.0, feedback=f"Invalid verification_info format: {err}")

        cases = ver_json.get("test_cases") if isinstance(ver_json, dict) else None
        if not cases:
            af.logger.trace("No test_cases in verification info.")
            return af.Evaluation(env=self, score=0.0, feedback="No public test cases available")

        passed, total = 0, len(cases)
        details: List[Dict[str, Any]] = []

        # Reuse one clean sandbox for all cases (faster; still isolated per-challenge)
        loop = asyncio.get_running_loop()
        with self._mgr.acquire() as box:
            # ensure working dir exists (tmpfs is set by manager)
            try:
                box.exec(["sh", "-lc", "mkdir -p /work && chmod 1777 /work"])
            except Exception as e:
                af.logger.warn(f"Failed to init workdir: {e}")

            for i, case in enumerate(cases, start=1):
                ctype = case.get("type")
                raw_inp = case.get("input")
                raw_exp = case.get("output")

                if ctype == "stdin_stdout":
                    inp = _to_str(raw_inp)
                    if not inp.endswith("\n"):
                        inp += "\n"
                    exec_prog = program
                    exp = _to_str(raw_exp)
                elif ctype == "function_call":
                    fn = case.get("fn_name")
                    args = case.get("input", [])
                    exec_prog = (
                        program
                        + "\n"
                        + f"if __name__ == '__main__':\n"
                        + f"    result = {fn}(*{args!r})\n"
                        + "    print(result)"
                    )
                    inp = ""
                    exp = _to_str(raw_exp[0]) if isinstance(raw_exp, list) and raw_exp else _to_str(raw_exp)
                else:
                    af.logger.trace(f"Unknown test case type '{ctype}', skipping.")
                    total -= 1
                    continue

                try:
                    out, err = await loop.run_in_executor(
                        None, self._executor.execute_in_lease, box, exec_prog, inp
                    )
                except Exception as ex:
                    out, err = "", f"EXEC_ERROR: {ex}"

                ok_run = not err.strip()  # stderr is merged; preserve legacy semantics
                out_norm = _normalize(out)
                exp_norm = _normalize(exp) if exp is not None else None
                correct = ok_run and (exp_norm is None or out_norm == exp_norm)
                if correct:
                    passed += 1
                    af.logger.trace(f"Test case {i} passed.")
                else:
                    af.logger.trace(f"Test case {i} failed. Got: {out_norm!r}, Expected: {exp_norm!r}")

                details.append(
                    {
                        "input": inp,
                        "expected": exp_norm,
                        "got": out_norm,
                        "stderr": err.strip(),
                        "passed": correct,
                    }
                )

        score = 1.0 if passed == total else 0.0
        feedback = json.dumps({"passed": passed, "total": total, "tests": details}, ensure_ascii=False)
        af.logger.trace(f"Evaluation done. Score={score}")
        return af.Evaluation(env=self, score=score, feedback=feedback)
