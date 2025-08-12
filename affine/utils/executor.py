from __future__ import annotations
from typing import Optional, Tuple

# Shared Docker-based program executor used by multiple envs
# Requires a SandboxLease-like object providing exec([...]) and put_file_bytes(path, bytes)

import re

DEFAULT_TIMEOUT_SEC   = 30
MAX_OUTPUT_BYTES      = 1_000_000
MAIN_PATH             = "/work/main.py"
STDIN_PATH            = "/work/stdin.txt"

_FENCE_RE  = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
_HAS_MAIN  = re.compile(r'if\s+__name__\s*==\s*[\'\"]__main__[\'\"]')

class ProgramExecutor:
    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        max_output: int = MAX_OUTPUT_BYTES,
        python_bin: str = "python3",
    ) -> None:
        self.timeout     = timeout
        self.max_output  = max_output
        self.python_bin  = python_bin
    @staticmethod
    def _strip_fences(text: str) -> str:
        m = _FENCE_RE.search(text or "")
        return (m.group(1) if m else (text or "")).strip()

    def _truncate_io(self, text: bytes | str, timed_out: bool) -> str:
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        if len(text.encode()) <= self.max_output:
            return text if not timed_out else (text + ("\n[TIMEOUT]" if not text.endswith("[TIMEOUT]") else ""))
        enc = text.encode()[: self.max_output]
        cut = enc.decode(errors="ignore") + "\n…<truncated>"
        if timed_out:
            cut += "\n[TIMEOUT]"
        return cut

    def _auto_runner_needed(self, src: str, out: str, err: str) -> bool:
        return (not out.strip() and not err.strip() and "def solve" in src and not _HAS_MAIN.search(src))

    def _write_files(self, box, code: str, stdin_data: str) -> None:
        program_payload = code if isinstance(code, str) else code.decode()
        script = (
            f"mkdir -p /work && chmod 1777 /work && \\\n"
            f"cat > {MAIN_PATH} << 'AFFINE_CODE'\n"
            f"{program_payload}\n"
            f"AFFINE_CODE\n"
            f"chmod 644 {MAIN_PATH}"
        )
        box.exec(["sh", "-lc", script])
        box.put_file_bytes(STDIN_PATH, (stdin_data or "").encode("utf-8"))

    def _exec_py(self, box, timeout: Optional[int], stdin_data: str) -> Tuple[int, bytes, bool]:
        tflag = f"timeout {int(timeout)}s " if timeout and timeout > 0 else ""
        payload = stdin_data if isinstance(stdin_data, str) else (stdin_data or "")
        if payload and not payload.endswith("\n"):
            payload += "\n"
        script = (
            f"{tflag}{self.python_bin} {MAIN_PATH} << 'AFFINE_EOF'\n"
            f"{payload}"
            f"AFFINE_EOF\n"
        )
        cmd = ["sh", "-lc", script]
        code, out = box.exec(cmd)
        timed_out = (code == 124 or code == 137)
        return code, out, timed_out

    def execute_in_lease(self, box, raw_code: str, stdin: str | bytes = "") -> Tuple[str, str]:
        code = self._strip_fences(raw_code)
        stdin_text = stdin if isinstance(stdin, str) else (stdin.decode() if isinstance(stdin, (bytes, bytearray)) else "")
        self._write_files(box, code, stdin_text)
        _, out_bytes, timed_out = self._exec_py(box, self.timeout, stdin_text)
        out_text = out_bytes.decode(errors="replace")
        err_text = ""
        if self._auto_runner_needed(code, out_text, err_text) and not timed_out:
            runner = (
                "\n\nif __name__ == \"__main__\":\n"
                "    res = solve()\n"
                "    if res is not None:\n"
                "        import sys\n"
                "        if isinstance(res, (list, tuple)):\n"
                "            print(*res)\n"
                "        else:\n"
                "            print(res)\n"
            )
            self._write_files(box, code + runner, stdin_text)
            _, out_bytes, timed_out = self._exec_py(box, self.timeout, stdin_text)
            out_text = out_bytes.decode(errors="replace")
        out_final = self._truncate_io(out_text, timed_out)
        err_final = ""
        return out_final, err_final 