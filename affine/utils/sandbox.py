import atexit
import io
import os
import tarfile
import time
import uuid
import weakref
from typing import Dict, Iterable, List, Optional, Tuple, Union

import docker


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


class SandboxLease:
    """
    A running container you can exec into multiple times.
    Cleans itself up on context exit, atexit, and GC via weakref.finalize.
    """

    def __init__(self, client: docker.DockerClient, container: docker.models.containers.Container, label_ns: str):
        self._client = client
        self._container = container
        self._label_ns = label_ns
        self._closed = False

        # GC fallback if user forgets to close
        self._finalizer = weakref.finalize(self, SandboxLease._finalize, client, container.id)

    @staticmethod
    def _finalize(client: docker.DockerClient, cid: str):
        try:
            c = client.containers.get(cid)
        except Exception:
            return
        try:
            if c.status != "exited":
                c.kill()
        except Exception:
            pass
        try:
            c.remove(force=True)
        except Exception:
            pass

    @property
    def id(self) -> str:
        return self._container.id

    def exec(
        self,
        cmd: Union[str, List[str]],
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
    ) -> Union[Tuple[int, bytes], Iterable[bytes]]:
        """
        Run a command inside the running container.
        Returns (exit_code, combined_output) by default.
        If stream=True, returns an iterator of log chunks (bytes).
        """
        exec_id = self._client.api.exec_create(
            container=self._container.id,
            cmd=cmd,
            workdir=workdir,
            environment=env,
            user=user,
            stdout=True,
            stderr=True,
        )["Id"]

        if stream:
            return self._client.api.exec_start(exec_id, stream=True)

        output = self._client.api.exec_start(exec_id, stream=False, detach=False, tty=False)
        # Optionally enforce timeout by polling exec_inspect, but docker-py blocks; simple case first:
        exit_code = self._client.api.exec_inspect(exec_id)["ExitCode"]
        return exit_code, output

    # ---------- File helpers ----------

    def put_file_bytes(self, container_path: str, data: bytes, mode: int = 0o644):
        parent = os.path.dirname(container_path).lstrip("/") or "."
        name = os.path.basename(container_path)
        tar_bytes = _single_file_tar_bytes(name, data, mode=mode)
        self._container.put_archive(path="/" + parent, data=tar_bytes)

    def put_path(self, src_path: str, dest_dir: str = "/work"):
        tar_bytes = _make_tar_bytes(src_path)
        self._container.put_archive(path=dest_dir, data=tar_bytes)

    def get_file_bytes(self, container_path: str) -> bytes:
        bits, _ = self._container.get_archive(container_path)
        raw = io.BytesIO()
        for chunk in bits:
            raw.write(chunk)
        raw.seek(0)
        with tarfile.open(fileobj=raw, mode="r:*") as tar:
            members = tar.getmembers()
            if not members:
                return b""
            f = members[0]
            return tar.extractfile(f).read() if f.isfile() else b""

    # ---------- Lifecycle ----------

    def reset(self):
        """Destroy and recreate the container with the same config (labels keep discovery simple)."""
        if self._closed:
            raise RuntimeError("Lease already closed.")
        cfg = self._container.attrs
        image = cfg["Config"]["Image"]
        labels = cfg["Config"].get("Labels", {}) or {}
        working_dir = cfg["Config"].get("WorkingDir") or "/work"
        # Remove old
        self.close()
        # Create new
        client = self._client
        container = client.containers.create(
            image=image,
            command=["sleep", "infinity"],
            detach=True,
            tty=False,
            stdin_open=False,
            working_dir=working_dir,
            labels=labels,
        )
        container.start()
        self._container = container
        # re-arm finalizer to new container id
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, SandboxLease._finalize, client, container.id)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._finalizer()  # run finalizer now (idempotent)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don’t suppress exceptions
        

class SandboxManager:
    """
    Factory for clean, resource-limited, labeled containers.
    Uses labels + a per-process session ID to allow a janitor to reap orphans at exit.
    """

    def __init__(
        self,
        image: str = "alpine:3.20",
        base_url: str = "unix:///var/run/docker.sock",
        pull: bool = True,
        workdir: str = "/work",
        tmpfs: Optional[Dict[str, str]] = None,         # e.g., {"/work": "rw,size=512m", "/tmp": "rw,size=256m"}
        volumes: Optional[Dict[str, Dict[str, str]]] = None,  # docker SDK mapping
        env: Optional[Dict[str, str]] = None,
        network_disabled: bool = True,
        read_only_root: bool = False,
        mem_limit: Optional[str] = None,                # e.g., "1g"
        cpus: Optional[float] = None,                   # e.g., 1.0
        label_ns: str = "rl-sandbox",
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
        self.cpus = cpus
        self.nano_cpus = _nano_cpus_from_float(cpus)
        self.label_ns = label_ns
        self.session_id = str(uuid.uuid4())

        if self.pull:
            self.client.images.pull(self.image)

        atexit.register(self._janitor)

    def acquire(self) -> SandboxLease:
        """Create and start a new 'clean' container that sleeps forever until you exec in it."""
        labels = {
            f"{self.label_ns}.managed": "1",
            f"{self.label_ns}.session": self.session_id,
        }
        container = self.client.containers.create(
            image=self.image,
            command=["sleep", "infinity"],
            detach=True,
            tty=False,
            stdin_open=False,
            working_dir=self.workdir,
            environment=self.env,
            labels=labels,
            network_disabled=self.network_disabled,
            read_only=self.read_only_root,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            volumes=self.volumes,
            tmpfs=self.tmpfs,
        )
        container.start()
        return SandboxLease(self.client, container, self.label_ns)

    def _janitor(self):
        """Best-effort cleanup of containers from this process session."""
        try:
            filters = {"label": [f"{self.label_ns}.managed=1", f"{self.label_ns}.session={self.session_id}"]}
            for c in self.client.containers.list(all=True, filters=filters):
                try:
                    if c.status != "exited":
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
