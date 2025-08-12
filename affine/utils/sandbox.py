# sandbox.py
from __future__ import annotations
import shlex
import atexit, io, os, tarfile, time, uuid, weakref, random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import docker
from docker.errors import APIError, NotFound


# ------------------------------- Exceptions -------------------------------- #

class SandboxError(Exception): ...
class SandboxNotRunning(SandboxError): ...
class SandboxExecError(SandboxError): ...
class SandboxTimeout(SandboxError): ...


# ------------------------------ Small helpers ------------------------------ #

def _nano_cpus_from_float(cpus: Optional[float]) -> Optional[int]:
    return None if cpus is None else int(cpus * 1e9)

def _make_tar_bytes(src_path: str, arcname: Optional[str] = None) -> bytes:
    """Create a tar stream containing src_path (file or directory)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(src_path, arcname=arcname or os.path.basename(src_path))
    buf.seek(0)
    return buf.read()

def _single_file_tar_bytes(name: str, data: bytes, mode: int = 0o644) -> bytes:
    """Create a tar stream containing a single file named `name` with contents `data`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = mode
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf.read()

def _is_running(container) -> bool:
    container.reload()
    st = container.attrs.get("State", {})
    return bool(st.get("Running")) and not st.get("Paused")

def _wait_running(container, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_running(container):
            return True
        time.sleep(0.05)
    return False

def _rand_backoff(attempt: int, base: float = 0.1, cap: float = 1.0):
    # jittered exponential backoff
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random())


# --------------------------------- Results --------------------------------- #

@dataclass
class ExecResult:
    code: int
    stdout: bytes
    stderr: bytes

    def check(self) -> "ExecResult":
        if self.code != 0:
            raise SandboxExecError(f"non-zero exit ({self.code})\nSTDERR:\n{self.stderr.decode(errors='ignore')}")
        return self

    @property
    def out(self) -> str:
        return self.stdout.decode(errors="ignore")

    @property
    def err(self) -> str:
        return self.stderr.decode(errors="ignore")


# --------------------------------- Lease ----------------------------------- #

class SandboxLease:
    """
    A running container you can exec into multiple times.
    Cleans itself up on context exit, atexit, and GC via weakref.finalize.
    """

    def __init__(
        self,
        client: docker.DockerClient,
        container: docker.models.containers.Container,
        label_ns: str,
        workdir: str,
        create_kwargs: Dict,          # used by reset()
        max_exec_retries: int = 5,
    ):
        self._client = client
        self._container = container
        self._label_ns = label_ns
        self._workdir = workdir
        self._create_kwargs = create_kwargs
        self._max_exec_retries = max_exec_retries
        self._closed = False

        # GC fallback if user forgets to close
        self._finalizer = weakref.finalize(self, SandboxLease._finalize, client, container.id)

    # -------------- Finalizer --------------

    @staticmethod
    def _finalize(client: docker.DockerClient, cid: str):
        try:
            c = client.containers.get(cid)
        except Exception:
            return
        try:
            c.kill()
        except Exception:
            pass
        try:
            c.remove(force=True)
        except Exception:
            pass

    # -------------- Properties --------------

    @property
    def id(self) -> str:
        return self._container.id

    @property
    def workdir(self) -> str:
        return self._workdir

    # -------------- Lifecycle --------------

    def _ensure_alive(self) -> None:
        """Start container if not running; wait until running."""
        try:
            if not _is_running(self._container):
                self._container.start()
                if not _wait_running(self._container, timeout=5.0):
                    raise SandboxNotRunning("container failed to reach Running state after start()")
        except NotFound:
            # Container was removed externally; recreate it
            self._recreate()

    def _recreate(self):
        """Recreate the container with original create kwargs."""
        if self._closed:
            raise SandboxError("lease is closed")
        # Kill and remove if it still exists
        try:
            self._container.kill()
        except Exception:
            pass
        try:
            self._container.remove(force=True)
        except Exception:
            pass
        # Create fresh
        container = self._client.containers.create(**self._create_kwargs)
        container.start()
        if not _wait_running(container, timeout=5.0):
            raise SandboxNotRunning("recreated container did not become Running")
        # prepare workdir
        self._mkdir_p(self._workdir, must_exist=True)
        # swap handle + finalizer id
        self._container = container
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, SandboxLease._finalize, self._client, container.id)

    def reset(self):
        """Destroy and recreate the container with the same config."""
        self._recreate()

    def close(self):
        if self._closed:
            return
        self._closed = True
        # run finalizer now (idempotent)
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don’t suppress exceptions

    # -------------- Exec --------------

    def exec(
        self,
        cmd: Union[str, List[str]],
        *,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        stream: bool = False,
        shell: bool = True,          # <- default to shell execution
        check: bool = False,
    ) -> Union[ExecResult, Iterator[bytes]]:
        """
        Run a command inside the container.

        Defaults to shell mode: we `cd` into the workdir in-shell to avoid Docker's
        own workdir chdir (which can intermittently fail on tmpfs and yield exit 128).
        """
        run_wd = workdir or self._workdir

        # Build command
        if shell:
            if isinstance(cmd, list):
                cmd = " ".join(shlex.quote(str(x)) for x in cmd)
            # cd inside the shell, do NOT use exec_create(workdir=...)
            cmd = ["/bin/sh", "-lc", f"cd {shlex.quote(run_wd)} && {cmd}"]
            exec_workdir = None
        else:
            # raw mode: still safer to avoid daemon workdir if you can
            exec_workdir = run_wd

        for attempt in range(self._max_exec_retries):
            self._ensure_alive()
            try:
                exec_id = self._client.api.exec_create(
                    container=self._container.id,
                    cmd=cmd,
                    workdir=exec_workdir,     # <- None in shell mode
                    environment=env,
                    user=user,
                    stdout=True,
                    stderr=True,
                )["Id"]

                if stream:
                    return self._client.api.exec_start(exec_id, stream=True, demux=False)

                out = self._client.api.exec_start(exec_id, stream=False, demux=True)
                insp = self._client.api.exec_inspect(exec_id)
                code = insp.get("ExitCode", 1)
                stdout, stderr = out if isinstance(out, tuple) else (out, b"")
                res = ExecResult(code=code, stdout=stdout or b"", stderr=stderr or b"")
                return res.check() if check else res

            except APIError as e:
                msg = str(e).lower()
                if any(s in msg for s in ["not running", "exec failed", "broken pipe",
                                        "context deadline exceeded", "transport is closing"]):
                    time.sleep(_rand_backoff(attempt))
                    continue
                raise

        self._container.reload()
        st = self._container.attrs.get("State", {})
        raise SandboxNotRunning(f"exec retries exhausted; state={st}")

    def sh(self, command: str, **kw) -> ExecResult:
        """Convenience: run a shell command via /bin/sh -lc."""
        return self.exec(command, shell=True, **kw)  # type: ignore[return-value]

    # -------------- FS helpers --------------

    def _mkdir_p(self, path: str, must_exist: bool = False):
        try:
            e_id = self._client.api.exec_create(self._container.id, cmd=["mkdir", "-p", path])["Id"]
            self._client.api.exec_start(e_id)
        except APIError:
            if must_exist:
                raise

    def put_file_bytes(self, container_path: str, data: bytes, mode: int = 0o644):
        parent = os.path.dirname(container_path).lstrip("/") or "."
        name = os.path.basename(container_path)
        self._mkdir_p("/" + parent)
        tar_bytes = _single_file_tar_bytes(name, data, mode=mode)
        self._container.put_archive(path="/" + parent, data=tar_bytes)

    def put_text(self, container_path: str, text: str, mode: int = 0o644, encoding="utf-8"):
        self.put_file_bytes(container_path, text.encode(encoding), mode=mode)

    def put_path(self, src_path: str, dest_dir: str = "/work"):
        self._mkdir_p(dest_dir)
        tar_bytes = _make_tar_bytes(src_path)
        self._container.put_archive(path=dest_dir, data=tar_bytes)

    def get_file_bytes(self, container_path: str) -> bytes:
        bits, _ = self._container.get_archive(container_path)
        raw = io.BytesIO()
        for chunk in bits:
            raw.write(chunk)
        raw.seek(0)
        with tarfile.open(fileobj=raw, mode="r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            if not members:
                return b""
            return tar.extractfile(members[0]).read()  # type: ignore[return-value]

    def get_text(self, container_path: str, encoding="utf-8") -> str:
        return self.get_file_bytes(container_path).decode(encoding, errors="ignore")

    # -------------- Introspection --------------

    def logs(self, tail: int = 200) -> str:
        try:
            return self._container.logs(tail=tail).decode(errors="ignore")
        except Exception:
            return ""

    def stats(self) -> Dict:
        self._container.reload()
        st = self._container.attrs.get("State", {})
        return {
            "running": st.get("Running"),
            "status": st.get("Status"),
            "exit_code": st.get("ExitCode"),
            "oom_killed": st.get("OOMKilled"),
            "pid": st.get("Pid"),
            "error": st.get("Error"),
        }


# -------------------------------- Manager ---------------------------------- #

class SandboxManager:
    """
    Factory for clean, resource-limited, labeled containers.
    Uses labels + a per-process session ID so a janitor can reap orphans at exit.
    """

    def __init__(
        self,
        image: str = "alpine:3.20",
        *,
        base_url: str = "unix:///var/run/docker.sock",
        pull: bool = True,
        workdir: str = "/work",
        tmpfs: Optional[Dict[str, str]] = None,          # {"/work": "rw,size=512m", "/tmp": "rw,size=256m"}
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        env: Optional[Dict[str, str]] = None,
        network_disabled: bool = True,
        read_only_root: bool = False,
        mem_limit: Optional[str] = None,                 # "1g"
        cpus: Optional[float] = None,                    # 1.0
        pids_limit: Optional[int] = None,                # e.g., 256
        label_ns: str = "rl-sandbox",
        max_exec_retries: int = 5,
        healthcheck_cmd: Optional[str] = None,           # e.g., "test -x /bin/sh || exit 1"
    ):
        self.client = docker.DockerClient(base_url=base_url)
        self.image = image
        self.pull = pull
        self.workdir = workdir
        self.tmpfs = tmpfs or {"/work": "rw,size=512m", "/tmp": "rw,size=256m"}
        self.volumes = volumes or {}
        self.env = env or {}
        self.network_disabled = network_disabled
        self.read_only_root = read_only_root
        self.mem_limit = mem_limit
        self.nano_cpus = _nano_cpus_from_float(cpus)
        self.pids_limit = pids_limit
        self.label_ns = label_ns
        self.session_id = str(uuid.uuid4())
        self.max_exec_retries = max_exec_retries
        self.healthcheck_cmd = healthcheck_cmd

        if self.pull:
            # Pull once up front to avoid a start/exec race during lazy pulls
            self.client.images.pull(self.image)

        atexit.register(self._janitor)

    # -------------- Create / Acquire --------------
    def _create_container(self, labels: Dict[str, str]) -> docker.models.containers.Container:
        """
        Create and start a container that stays alive indefinitely.
        We avoid any exec() during creation to eliminate 409 'not running' races.
        /work is provided via tmpfs with mode=1777 so non-root users can write.
        """
        # --- Healthcheck (optional) ---
        healthcheck = None
        if self.healthcheck_cmd:
            to_ns = lambda s: int(s * 1_000_000_000)
            healthcheck = {
                "Test": ["CMD-SHELL", self.healthcheck_cmd],
                "Interval": to_ns(1.0),
                "Timeout": to_ns(2.0),
                "Retries": 3,
                "StartPeriod": to_ns(1.0),
            }

        # --- Ensure tmpfs options are safe for non-root (sticky world-writable) ---
        tmpfs = dict(self.tmpfs or {})
        def _ensure_mode(path: str):
            if path in tmpfs:
                # Append mode=1777 if not explicitly set
                if "mode=" not in tmpfs[path]:
                    tmpfs[path] = tmpfs[path] + ("," if tmpfs[path] else "") + "mode=1777"

        _ensure_mode(self.workdir)      # e.g. "/work"
        _ensure_mode("/tmp")            # keep /tmp safe too

        # --- Build create kwargs ---
        create_kwargs = dict(
            image=self.image,
            # Avoid shell dependency for PID 1; 'tail -f /dev/null' is rock-solid
            entrypoint=["tail", "-f", "/dev/null"],
            command=None,
            init=True,
            detach=True,
            tty=False,
            stdin_open=False,
            working_dir="/",                    # don't start in /work (mount timing)
            environment=self.env,
            labels=labels,
            network_disabled=self.network_disabled,
            read_only=self.read_only_root,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            pids_limit=self.pids_limit,
            volumes=self.volumes,
            tmpfs=tmpfs,                        # includes mode=1777 for /work,/tmp
            healthcheck=healthcheck,
        )

        # --- Create + start ---
        container = self.client.containers.create(**create_kwargs)
        container.start()

        # Wait until the daemon reports Running
        if not _wait_running(container, timeout=5.0):
            container.reload()
            st = container.attrs.get("State", {})
            try:
                logs = container.logs(tail=100).decode(errors="ignore")
            except Exception:
                logs = ""
            try:
                container.remove(force=True)
            except Exception:
                pass
            raise SandboxNotRunning(f"container failed to start; state={st}; logs:\n{logs}")

        # Save kwargs for reset()/recreate()
        container._sandbox_create_kwargs = create_kwargs  # type: ignore[attr-defined]
        return container


    def acquire(self) -> SandboxLease:
        """Create and start a new 'clean' container that sleeps forever until you exec in it."""
        labels = {
            f"{self.label_ns}.managed": "1",
            f"{self.label_ns}.session": self.session_id,
        }
        container = self._create_container(labels=labels)
        lease = SandboxLease(
            client=self.client,
            container=container,
            label_ns=self.label_ns,
            workdir=self.workdir,
            create_kwargs=container._sandbox_create_kwargs,  # type: ignore[attr-defined]
            max_exec_retries=self.max_exec_retries,
        )
        return lease

    # -------------- Janitor --------------

    def _janitor(self):
        """Best-effort cleanup of containers from this process session."""
        try:
            filters = {"label": [f"{self.label_ns}.managed=1", f"{self.label_ns}.session={self.session_id}"]}
            for c in self.client.containers.list(all=True, filters=filters):
                try:
                    c.kill()
                except Exception:
                    pass
                try:
                    c.remove(force=True)
                except Exception:
                    pass
        except Exception:
            # Silent best effort
            pass

    def reap_all(self):
        """Kill/remove ALL containers for this label namespace, regardless of session."""
        try:
            filters = {"label": [f"{self.label_ns}.managed=1"]}
            for c in self.client.containers.list(all=True, filters=filters):
                try:
                    c.kill()
                except Exception:
                    pass
                try:
                    c.remove(force=True)
                except Exception:
                    pass
        except Exception:
            pass
