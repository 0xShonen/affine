
#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import os
import re
import sys
import math
import json
import time
import click
import socket
import random
import hashlib
import aiohttp
import asyncio
import logging
import requests
import textwrap
import traceback
import itertools
from .utils import *
from math import comb
import datetime as dt
from tqdm import tqdm
import bittensor as bt
import datasets as hf_ds                    
from pathlib import Path
from types import NoneType
from tqdm.asyncio import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
from typing import AsyncIterator
from urllib.parse import urlparse
from huggingface_hub import HfApi
from botocore.config import Config
from collections import defaultdict
from abc import ABC, abstractmethod
from pydantic import root_validator
from aiohttp import ClientConnectorError
from aiobotocore.session import get_session
from huggingface_hub import snapshot_download
from bittensor.core.errors import MetadataError
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable
__version__ = "0.0.0"

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
NETUID = 120
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")
_SINGLETON_CACHE = {}
def singleton(key:str, factory):
    """Create a singleton factory function that creates an object only once."""
    def get_instance():
        if key not in _SINGLETON_CACHE:
            _SINGLETON_CACHE[key] = factory()
        return _SINGLETON_CACHE[key]
    return get_instance

# --------------------------------------------------------------------------- #
#                       Prometheus                         #
# --------------------------------------------------------------------------- #
from prometheus_client import Counter, CollectorRegistry, start_http_server, Gauge
METRICS_PORT   = int(os.getenv("AFFINE_METRICS_PORT", "8000"))
METRICS_ADDR   = os.getenv("AFFINE_METRICS_ADDR", "0.0.0.0")
REGISTRY       = CollectorRegistry(auto_describe=True)
QCOUNT  = Counter("qcount", "qcount", ["model"], registry=REGISTRY)
SCORE   = Gauge( "score", "score", ["uid", "env"], registry=REGISTRY)
RANK    = Gauge( "rank", "rank", ["uid", "env"], registry=REGISTRY)
WEIGHT  = Gauge( "weight", "weight", ["uid"], registry=REGISTRY)
LASTSET = Gauge( "lastset", "lastset", registry=REGISTRY)
NRESULTS = Gauge( "nresults", "nresults", registry=REGISTRY)
MAXENV = Gauge("maxenv", "maxenv", ["env"], registry=REGISTRY)
CACHE = Gauge( "cache", "cache", registry=REGISTRY)

# Model gating check cache
MODEL_GATING_CACHE = {}  # {model_id: (is_gated, last_checked)}
# Replace global loop-bound lock with per-event-loop lazy locks to avoid cross-loop errors
_GATING_LOCKS: Dict[int, asyncio.Lock] = {}
GATING_TTL = 300  # 5 minutes

def _get_gating_lock() -> asyncio.Lock:
    """Return an asyncio.Lock bound to the current running loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Fallback if called when no loop is running yet
        loop = asyncio.get_event_loop()
    key = id(loop)
    lock = _GATING_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _GATING_LOCKS[key] = lock
    return lock

# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    if not getattr(setup_logging, "_prom_started", False):
        try: start_http_server(METRICS_PORT, METRICS_ADDR, registry=REGISTRY)
        except: pass
        setup_logging._prom_started = True
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
def info():setup_logging(1)
def debug():setup_logging(2)
def trace():setup_logging(3)

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key); 
    if not v and default is None:
        raise ValueError(f"{key} not set.\nYou must set env var: {key} in .env")
    return v or default

async def check_model_gated(model_id: str, revision: Optional[str] = None) -> Optional[bool]:
    async with _get_gating_lock():
        now = time.time()
        cached = MODEL_GATING_CACHE.get(model_id)
        if cached and now - cached[1] < GATING_TTL:
            return cached[0]
        try:
            r = await asyncio.to_thread(requests.get, f"https://huggingface.co/api/models/{model_id}", timeout=5)
            if r.status_code == 200:
                is_gated = r.json().get("gated", False)
                if revision:
                    try:
                        ok = await asyncio.to_thread(lambda: bool(HfApi(token=os.getenv("HF_TOKEN")).repo_info(repo_id=model_id, revision=revision, repo_type="model")))
                        if not ok: is_gated = True
                    except:
                        pass
                MODEL_GATING_CACHE[model_id] = (is_gated, now)
                return is_gated
        except Exception as e:
            logger.trace(f"Gate check failed for {model_id}: {e}")
        if cached:
            MODEL_GATING_CACHE[model_id] = (cached[0], now)
            return cached[0]
        return None


# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.trace("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor( get_conf('SUBTENSOR_ENDPOINT', default='finney') )
        try:
            await SUBTENSOR.initialize()
            logger.trace("Connected")
        except Exception as e:
            logger.warning(f"Failed to initialize subtensor: {e}, falling back to {'wss://lite.sub.latent.to:443'}")
            SUBTENSOR = bt.async_subtensor( get_conf('SUBTENSOR_FALLBACK', default="wss://lite.sub.latent.to:443") )
            await SUBTENSOR.initialize()
            logger.trace("Connected to fallback")
    return SUBTENSOR

# --------------------------------------------------------------------------- #
#                           Base‑level data models                            #
# --------------------------------------------------------------------------- #
def _truncate(t: Optional[str], max_len: int = 80) -> str:
    return "" if not t else textwrap.shorten(t, width=max_len, placeholder="…")

class BaseEnv(BaseModel, ABC):
    __version__: str = "0.0.0"
    class Config: arbitrary_types_allowed = True
    @property
    def key(self) -> str:   return "E-" + str(self.__class__.__name__) + "-" + str(self.__version__)
    @property
    def name(self) -> str:  return self.key
    def __repr__(self):     return self.key
    def __str__(self):     return self.key
    def __hash__(self):     return hash(self.key)
    @abstractmethod
    async def generate(self) -> "Challenge": ...
    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: "Response") -> "Evaluation": ...

# --------------------------------------------------------------------------- #
#                         Models with new (de)serialisation                   #
# --------------------------------------------------------------------------- #
class Challenge(BaseModel):
    env:  BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    challenge_id: Optional[str] = None
    @root_validator(pre=True)
    def set_challenge_id(cls, values):
        if "challenge_id" not in values or values["challenge_id"] is None:
            env = values["env"]
            prompt = values["prompt"]
            extra = values.get("extra", {})
            if not isinstance(env, str): env = env.name
            base_dict = { "env": env,"prompt": prompt, "extra": extra}
            canonical = json.dumps(base_dict, sort_keys=True, separators=(",", ":"))
            cid = hashlib.sha256(canonical.encode()).hexdigest()
            values["challenge_id"] = cid
        return values
    @validator("env", pre=True)
    def _parse_env(cls, v):
        from .envs import ENVS as _ENVS
        return _ENVS[v]() if isinstance(v, str) else v
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    async def evaluate(self, resp: "Response") -> "Evaluation":
        return await self.env.evaluate(self, resp)
    def __repr__(self):
        return f"<Challenge env={self.env.name!r} prompt={_truncate(self.prompt)!r}>"
    __str__ = __repr__


class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)
    @validator("env", pre=True)
    def _parse_env(cls, v):
        from .envs import ENVS as _ENVS
        return _ENVS[v]() if isinstance(v, str) else v
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    def __repr__(self):
        ex = {k: _truncate(str(v)) for k, v in self.extra.items()}
        return f"<Evaluation env={self.env.name!r} score={self.score:.4f} extra={ex!r}>"
    __str__ = __repr__

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]
    success: bool
    def __repr__(self):
        return (f"<Response model={self.model!r} success={self.success} "
                f"latency={self.latency_seconds:.3f}s attempts={self.attempts} "
                f"response={_truncate(self.response)!r} error={_truncate(self.error)!r}>")
    __str__ = __repr__

class Miner(BaseModel):
    uid: int; hotkey: str; model: Optional[str] = None
    revision: Optional[str] = None; block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None
    slug: Optional[str] = None
    @property
    def key(self) -> str:   
        key_str = "MNR-" + "R" + str(self.revision if self.revision else "None") 
        key_str += "-M" + str(self.model if self.model else "None") 
        key_str += "-U" + str(self.uid if self.uid else "None") 
        key_str += "-B" + str(self.block if self.block else "None")
        return str(key_str)
    def __hash__(self):     return hash(self.key)
    def __repr__(self):     return str(self.key)
    def __str__(self):      return str(self.key)

class Result(BaseModel):
    version: str = __version__
    signature: str = ""
    hotkey: str = ""
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation
    def sign(self, wallet):
        self.hotkey = wallet.hotkey.ss58_address
        self.signature = (wallet.hotkey.sign( data = str(self.challenge) )).hex()
    def verify( self ) -> bool:
        return bt.Keypair(ss58_address=self.hotkey).verify( data = str(self.challenge), signature = bytes.fromhex( self.signature) )
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    def __repr__(self): return f"<Result {self.miner.uid=} {self.challenge.env.name=} score={self.evaluation.score:.4f}>"
    __str__ = __repr__

# Central env registry
from .envs import ENVS


# --------------------------------------------------------------------------- #
#                               QUERY                                         #
# --------------------------------------------------------------------------- #
# Lazy-initialised semaphore and shared HTTP client
_HTTP_SEMS: Dict[int, asyncio.Semaphore] = {}
_CLIENTS: Dict[int, aiohttp.ClientSession] = {}

async def _get_sem() -> asyncio.Semaphore:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    sem = _HTTP_SEMS.get(key)
    if sem is None:
        sem = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
        _HTTP_SEMS[key] = sem
    return sem

async def _get_client() -> aiohttp.ClientSession:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    client = _CLIENTS.get(key)
    if client is None or client.closed:
        client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        _CLIENTS[key] = client
    return client

TERMINAL = {400, 404, 410}
async def query(prompt, model: str = "unsloth/gemma-3-12b-it", slug: str = "llm", timeout=150, retries=0, backoff=1) -> Response:
    url = f"https://{slug}.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
    start = time.monotonic()
    QCOUNT.labels(model=model).inc()
    R = lambda resp, at, err, ok: Response(response=resp, latency_seconds=time.monotonic()-start,
                                          attempts=at, model=model, error=err, success=ok)
    sess = await _get_client()
    sem = await _get_sem()
    for attempt in range(1, retries+2):
        try:
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            async with sem, sess.post(url, json=payload,
                                      headers=hdr, timeout=timeout) as r:
                    txt = await r.text(errors="ignore")
                    if r.status in TERMINAL: return R(None, attempt, f"{r.status}:{txt}", False)
                    r.raise_for_status()
                    content = (await r.json())["choices"][0]["message"]["content"]
                    return R(content, attempt, None, True)
        except Exception as e:
            if attempt > retries: return R(None, attempt, str(e), False)
            await asyncio.sleep(backoff * 2**(attempt-1) * (1 + random.uniform(-0.1, 0.1)))

LOG_TEMPLATE = (
    "[RESULT] "
    "{pct:>3.0f}% | "
    "U{uid:>3d} │ "
    "{model:<50s} │ "
    "{env:<3} │ "
    "{success:^4s} │ "
    "{score:>6.4f} │ "
    "{latency:>6.3f}s"
)
async def run(challenges, miners, timeout=240, retries=0, backoff=1 )-> List[Result]:
    if not isinstance(challenges, list): challenges = [challenges]
    if isinstance(miners, Miner): miners = [miners]
    if isinstance(miners, dict):  mmap = miners
    elif isinstance(miners, list) and all(hasattr(m, "uid") for m in miners):  mmap = {m.uid: m for m in miners}
    else: mmap = await miners(miners)
    
    logger.trace("Running challenges: %s on miners: %s", [chal.prompt[:30] for chal in challenges], list(mmap.keys()))
    response = []
    
    async def proc(miner, chal):
        # Check gating status before querying
        if miner.model:
            is_gated = await check_model_gated(miner.model, miner.revision)
            if is_gated is True:
                err = "Model is gated"
                logger.trace(f"Miner {miner.uid} - {err} for model {miner.model}")
                resp = Response(response=None, latency_seconds=0, attempts=0, model=miner.model, error=err, success=False)
                ev = Evaluation(env=chal.env, score=0.0, extra={"error": err, "gated": is_gated})
                return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)
        
        # Normal processing for non-gated models
        resp = await query(chal.prompt, miner.model, miner.slug, timeout, retries, backoff)
        try: ev = await chal.evaluate(resp)
        except Exception as e: ev = Evaluation(env=chal.env, score=0.0, extra={"error": str(e), "evaluation_failed": True})
        return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)
    
    tasks = [ asyncio.create_task(proc(m, chal)) for m in mmap.values() if m.model for chal in challenges]  
    total = len(tasks); completed = 0
    for task in asyncio.as_completed(tasks): 
        result: Result = await task
        response.append(result); completed += 1
        logger.debug(
            LOG_TEMPLATE.format(
                pct    = completed / total * 100,
                env    = result.challenge.env.name,                   
                uid    = result.miner.uid,                 
                model  = result.miner.model[:50] or "",         
                success= "RECV" if result.response.success else "NULL",
                score  = result.evaluation.score,
                latency= result.response.latency_seconds
            )
        )
    return response


# --------------------------------------------------------------------------- #
#                              Miners                                         #
# --------------------------------------------------------------------------- #
async def get_chute(chutes_id: str) -> Dict[str, Any]:
    url = f"https://api.chutes.ai/chutes/{chutes_id}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            text = await r.text(errors="ignore")
            if r.status != 200:
                return None
            info = await r.json()
            for k in ('readme','cords','tagline','instances'):
                info.pop(k, None)
            info.get('image', {}).pop('readme', None)
            return info
        
async def get_chute_code(identifier: str) -> Optional[str]:
    url = f"https://api.chutes.ai/chutes/code/{identifier}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            return await r.text(errors="ignore")

async def get_latest_chute_id(model_name: str, api_key: Optional[str] = None) -> Optional[str]:
    token = api_key or os.getenv("CHUTES_API_KEY", ""); 
    if not token: return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.chutes.ai/chutes/", headers={"Authorization": token}) as r:
                if r.status != 200: return None
                data = await r.json()
    except Exception: return None
    chutes = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(chutes, list): return None
    for chute in reversed(chutes):
        if any(chute.get(k) == model_name for k in ("model_name", "name", "readme")):
            return chute.get("chute_id")
    return None


async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: object = None,
) -> Dict[int, Miner]:
    sub = await get_subtensor()
    meta = meta or await sub.metagraph(netuid)
    commits = await sub.get_all_revealed_commitments(netuid)
    if uids is None:uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int): uids = [uids]    
    async def fetch(uid: int):
        try:
            hotkey = meta.hotkeys[ uid ]
            if hotkey not in commits: return None
            commit = commits[hotkey]
            block, data = commit[-1]     
            block = 0 if uid == 0 else block
            data = json.loads(data)
            model, miner_revision, chute_id = data.get("model"), data.get("revision"), data.get("chute_id")
            chute = await get_chute(chute_id)
            if not chute: None
            gated = await check_model_gated(model)
            if gated: return None
            chutes_name, slug, chutes_revision = chute.get('name'), chute.get("slug"), chute.get("revision")
            if model != chutes_name or (uid != 0 and chutes_name.split('/')[1].lower()[:6] != 'affine'): return None
            if chutes_revision == None or miner_revision == chutes_revision:
                miner = Miner(
                    uid=uid, hotkey=hotkey, model=model, block=int(block),
                    revision = miner_revision,
                    slug = slug,
                    chute=chute,
                )
                return miner
        except: pass
    results = await asyncio.gather(*(fetch(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}
    # Remove duplicates.
    if output:
        best_by_model: Dict[str, Tuple[int, int]] = {}
        for uid, m in output.items():
            if not m.model:
                continue
            blk = m.block if isinstance(m.block, int) else (int(m.block) if m.block is not None else (2**63 - 1))
            prev = best_by_model.get(m.model)
            if prev is None or blk < prev[0]:
                best_by_model[m.model] = (blk, uid)
        selected_uids = {uid for _, uid in best_by_model.values()}
        output = {uid: m for uid, m in output.items() if uid in selected_uids}
    return output


# --------------------------------------------------------------------------- #
#                   S3 helpers                                                #
# --------------------------------------------------------------------------- #
# ── ENV ──────────────────────────────────────────────────────────────────────
WINDOW        = int(os.getenv("AFFINE_WINDOW", 20))
RESULT_PREFIX = "affine/results/"
INDEX_KEY     = "affine/index.json"

FOLDER  = os.getenv("R2_FOLDER", "affine" )
BUCKET  = os.getenv("R2_BUCKET_ID", "80f15715bb0b882c9e967c13e677ed7d" )
ACCESS  = os.getenv("R2_WRITE_ACCESS_KEY_ID", "ff3f4f078019b064bfb6347c270bee4d")
SECRET  = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "a94b20516013519b2959cbbb441b9d1ec8511dce3c248223d947be8e85ec754d")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

get_client_ctx = lambda: get_session().create_client(
    "s3", endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
    config=Config(max_pool_connections=256)
)

CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR",
                 Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── fast JSON ───────────────────────────────────────────────────────────────
try:
    import orjson as _json
    _loads, _dumps = _json.loads, _json.dumps
except ModuleNotFoundError:
    _loads = lambda b: json.loads(b.decode())
    _dumps = lambda o: json.dumps(o, separators=(",", ":")).encode()
    
# ── Index helpers ───────────────────────────────────────────────────────────
def _miner_prefix(env: "BaseEnv", miner: Miner) -> str:
    assert hasattr(env, "key"), "env must be a BaseEnv with a .key"
    assert hasattr(miner, "key"), "miner must be a Miner with a .key"
    return f"{RESULT_PREFIX}{env.key}/{miner.key}/"

def _miner_index_key(env: "BaseEnv", miner: Miner) -> str:
    return f"{_miner_prefix(env, miner)}index.json"

# Key-based internal helpers for string-only contexts (e.g., dataset/index scan)
def _miner_prefix_keys(env_key: str, miner_key: str) -> str:
    return f"{RESULT_PREFIX}{env_key}/{miner_key}/"

def _miner_index_key_keys(env_key: str, miner_key: str) -> str:
    return f"{_miner_prefix_keys(env_key, miner_key)}index.json"

# ── Index discovery ────────────────────────────────────────────────────────
async def _index() -> list[str]:
    """Return list of all miner index.json keys under RESULT_PREFIX."""
    async with get_client_ctx() as c:
        paginator = c.get_paginator("list_objects_v2")
        out = []
        async for page in paginator.paginate(Bucket=FOLDER, Prefix=RESULT_PREFIX):
            for o in page.get("Contents", []):
                if o["Key"].endswith("/index.json"):
                    out.append(o["Key"])
        return sorted(out)

async def _update_index(k: str) -> None:
    """Keep a global list of all index.json files at INDEX_KEY (optional)."""
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
            idx = set(json.loads(await r["Body"].read()))
        except c.exceptions.NoSuchKey:
            idx = set()
        if k not in idx:
            idx.add(k)
            await c.put_object(Bucket=FOLDER, Key=INDEX_KEY,
                               Body=_dumps(sorted(idx)),
                               ContentType="application/json")

# ── Head counters ──────────────────────────────────────────────────────────
async def get_head(env: "BaseEnv", miner: Miner) -> int:
    """Fast path via index.json; fallback to listing if missing."""
    assert hasattr(env, "key"), "env must be a BaseEnv with a .key"
    assert hasattr(miner, "key"), "miner must be a Miner with a .key"
    key = _miner_index_key(env, miner)
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=key)
            return int(json.loads(await r["Body"].read())["head"])
        except c.exceptions.NoSuchKey:
            # fallback: list existing COUNT.json and take max
            prefix = _miner_prefix(env, miner)
            paginator = c.get_paginator("list_objects_v2")
            head = 0
            async for page in paginator.paginate(Bucket=FOLDER, Prefix=prefix):
                for o in page.get("Contents", []):
                    n = Path(o["Key"]).stem
                    if n.isdigit(): head = max(head, int(n))
            return head

async def set_head(env: "BaseEnv", miner: Miner, head: int) -> None:
    assert hasattr(env, "key"), "env must be a BaseEnv with a .key"
    assert hasattr(miner, "key"), "miner must be a Miner with a .key"
    key = _miner_index_key(env, miner)
    async with get_client_ctx() as c:
        await c.put_object(Bucket=FOLDER, Key=key,
                           Body=_dumps({"head": int(head)}),
                           ContentType="application/json")
    await _update_index(key)

# ── Sink with signed results ───────────────────────────────────────────────────
async def sign_results( wallet, results ):
    try:
        signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
        timeout = aiohttp.ClientTimeout(connect=2, total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payloads = [str(r.challenge) for r in results]
            resp = await session.post(f"{signer_url}/sign", json={"payloads": payloads})
            if resp.status == 200:
                data = await resp.json()
                sigs = data.get("signatures") or []
                hotkey = data.get("hotkey")
                for r, s in zip(results, sigs):
                    r.hotkey = hotkey
                    r.signature = s
    except Exception as e:
        logger.info(f"sink: signer unavailable, using local signing: {type(e).__name__}: {e}")
        hotkey = wallet.hotkey.ss58_address
        for r in results: 
            r.sign(wallet)
    finally:
        return hotkey, results
    
async def sink(wallet: bt.wallet, env: "BaseEnv", miner: Miner, results: list["Result"]):
    if not results: return
    hotkey, signed = await sign_results(wallet, results)
    dumped = [r.model_dump(mode="json") for r in signed]

    assert hasattr(env, "key") and hasattr(miner, "key")
    head = await get_head(env, miner)
    new_head = head + len(dumped)
    key = f"{_miner_prefix(env, miner)}{new_head}.json"

    async with get_client_ctx() as c:
        await c.put_object(Bucket=FOLDER, Key=key,
                           Body=_dumps(dumped),
                           ContentType="application/json")
    await set_head(env, miner, new_head)

# ── Cache of COUNT.json batches (small, single JSON arrays) ────────────────
async def _cache_batch(key: str, sem: asyncio.Semaphore) -> Path:
    # Namespace cache by env/miner to avoid filename collisions across pairs
    rel = key[len(RESULT_PREFIX):] if key.startswith(RESULT_PREFIX) else key
    out = CACHE_DIR / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    mod = out.with_suffix(".modified")
    async with sem, get_client_ctx() as c:
        if out.exists() and mod.exists():
            h = await c.head_object(Bucket=FOLDER, Key=key)
            if h["LastModified"].isoformat() == mod.read_text().strip():
                return out
        o = await c.get_object(Bucket=FOLDER, Key=key)
        body, lm = await o["Body"].read(), o["LastModified"].isoformat()
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f: f.write(body)
    os.replace(tmp, out); mod.write_text(lm)
    return out

# ── Iterate compactly over stored Results ──────────────────────────────────
async def _iter_batch_file(p: Path):
    try:
        import aiofiles
        async with aiofiles.open(p, "rb") as f:
            data = _loads(await f.read())   # batch is a JSON array
    except ModuleNotFoundError:
        data = await asyncio.to_thread(lambda: _loads(p.read_bytes()))
    for obj in data:
        try:
            r = Result.model_validate(obj)
            if r.verify(): yield r
        except Exception:
            pass

# ── Core async stream (Result objects) ─────────────────────────────────────
async def dataset(
    tail: int,
    envs: list = None,
    miners: list = None,
    *,
    max_concurrency: int = 10,
) -> AsyncIterator["Result"]:
    """
    Stream up to `tail` recent results across selected (env, miner) paths.
    Order: increasing COUNT within each (env,miner); global interleave by LastModified.
    """
    # discover (env, miner, head)
    idx_keys = await _index()
    pairs = []
    async with get_client_ctx() as c:
        for ik in idx_keys:
            parts = Path(ik).parts  # ['affine','results','<env>','<miner>','index.json']
            if len(parts) < 5: continue
            env, miner = parts[-3], parts[-2]
            if envs:
                allowed_envs = { (getattr(e, "key", None) or getattr(e, "name", None) or str(e)) for e in envs }
                if (env not in allowed_envs):
                    continue
            if miners:
                allowed_miners = { (getattr(m, "key", None) or str(m)) for m in miners }
                if (miner not in allowed_miners):
                    continue
            try:
                r = await c.get_object(Bucket=FOLDER, Key=ik)
                head = int(json.loads(await r["Body"].read())["head"])
                if head > 0: pairs.append((env, miner, head))
            except Exception:
                pass

    # build list of COUNT.json keys (bounded by tail heuristic)
    keys = []
    remain = tail
    for env, miner, head in pairs:
        # pull up to `remain` from the end for each (env,miner); adjust as needed
        start = max(1, head - remain + 1)
        for n in range(start, head + 1):
            keys.append((env, miner, n))
    # fetch in order of LastModified (newest last for stable forward stream)
    async with get_client_ctx() as c:
        s3keys = [f"{_miner_prefix_keys(e,m)}{n}.json" for e,m,n in keys]
        lm = {}
        for k in s3keys[:1000]:  # cheap head for ordering; cap per call
            try:
                h = await c.head_object(Bucket=FOLDER, Key=k)
                lm[k] = h["LastModified"]
            except Exception:
                pass
    s3keys.sort(key=lambda k: lm.get(k, dt.datetime.min))

    # prefetch/cache and stream
    sem = asyncio.Semaphore(max_concurrency)
    files = [asyncio.create_task(_cache_batch(k, sem)) for k in s3keys]
    emitted = 0
    for t in files:
        p = await t
        async for r in _iter_batch_file(p):
            yield r
            emitted += 1
            if emitted >= tail: return

# ── Prune cache by COUNT threshold (simple heuristic) ──────────────────────
async def prune(tail: int):
    """
    Ensure exactly `tail` results are retained per (env, miner) cache.
    Deletes older files entirely when fully before the cutoff, and rewrites the
    boundary file to keep only the needed suffix so the retained total equals `tail`.
    """
    try:
        idx_keys = await _index()
    except Exception:
        idx_keys = []

    # Map env/miner -> (remote head, local dir)
    pairs: list[tuple[str, str, int, Path]] = []
    async with get_client_ctx() as c:
        for ik in idx_keys:
            parts = Path(ik).parts
            if len(parts) < 5:
                continue
            env_key, miner_key = parts[-3], parts[-2]
            try:
                r = await c.get_object(Bucket=FOLDER, Key=ik)
                head = int(json.loads(await r["Body"].read())["head"])
            except Exception:
                head = 0
            local_dir = CACHE_DIR / env_key / miner_key
            if head > 0 and local_dir.exists():
                pairs.append((env_key, miner_key, head, local_dir))

    # Prune each (env, miner)
    for _, _, head, local_dir in pairs:
        try:
            cutoff = max(0, head - tail)
            files = [p for p in local_dir.glob("*.json") if p.stem.isdigit()]
            files.sort(key=lambda p: int(p.stem))
            prev_head = 0
            for p in files:
                H = int(p.stem)
                if H <= cutoff:
                    try:
                        p.unlink()
                    except OSError:
                        pass
                    try:
                        p.with_suffix(".modified").unlink()
                    except OSError:
                        pass
                    prev_head = H
                    continue
                # Boundary file: prev_head <= cutoff < H
                if prev_head <= cutoff < H:
                    need = H - cutoff
                    try:
                        data = await asyncio.to_thread(lambda: _loads(p.read_bytes()))
                    except Exception:
                        data = []
                    if isinstance(data, list) and len(data) > need:
                        trimmed = data[-need:]
                        tmp = p.with_suffix(".tmp")
                        try:
                            with tmp.open("wb") as f:
                                f.write(_dumps(trimmed))
                            os.replace(tmp, p)
                        finally:
                            try:
                                if tmp.exists(): tmp.unlink()
                            except Exception:
                                pass
                prev_head = H
        except Exception:
            # best-effort per directory; continue
            continue


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)
    
# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)
            
# --------------------------------------------------------------------------- #
#                               Runner                                        #
# --------------------------------------------------------------------------- #
@cli.command("runner")
def runner():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    logger.info(f"Runner starting with wallet {coldkey}/{hotkey}")
    # from .envs.sat import SAT
    # envs = {SAT.__name__: SAT()}
    envs = {name: cls() for name, cls in ENVS.items()}   # single instances
    logger.info(f"Building envs: {envs}")

    async def _run():
        logger.debug("Warming up...")
        # Warm up subtensor.
        await get_subtensor()
        logger.info(f"Subtensor is warm.")

        # Warm up envs.
        for env in envs.values():
            await env.generate()
            logger.info(f"{env} is warm")
            await asyncio.sleep(3)
        logger.info(f"Envs are warm")

        MINERS = None
        WEIGHTS = None
        async def sync_state():
            nonlocal MINERS, WEIGHTS
            logger.info('Sync state ...')
            sub = await get_subtensor()
            meta = await sub.metagraph(NETUID)
            MINERS = await miners( meta = meta )  
            logger.info(f'Pulled {len(MINERS.keys())} miners')
            WEIGHTS = defaultdict(lambda: defaultdict(float))
            logger.info('Pulling counts ...')
            async def get_count_for_miner(uid, m):
                miner_counts = {}
                for _, e_inst in envs.items():
                    count = await get_head(e_inst, m)
                    miner_counts[e_inst.key] = count
                return m.key, miner_counts
            tasks = [get_count_for_miner(uid, m) for uid, m in MINERS.items()]
            results = await asyncio.gather(*tasks)
            counts = dict(results)
            eps = 1e-6
            env_keys = [e.key for e in envs.values()]
            for e_key in env_keys:
                table_data = [];
                emax = max([counts.get(m.key, {}).get(e_key, 0) for m in MINERS.values()]) if MINERS else 0
                etotal = sum([counts.get(m.key, {}).get(e_key, 0) for m in MINERS.values()]) if MINERS else 0
                for m in MINERS.values():
                    weight = (emax - counts.get(m.key, {}).get(e_key, 0) + eps) / (etotal + eps)
                    WEIGHTS[m.key][e_key] = weight
                    row = {'uid': m.uid, 'env': e_key, 'cnt': counts.get(m.key, {}).get(e_key, 0), 'max': emax, 'w': weight }
                    table_data.append(row)
                logger.info("\n" + tabulate(table_data, headers='keys', tablefmt='pretty'))
                
        # Create a env and query a single miner.
        K = 1
        BUFFER = defaultdict(lambda: defaultdict(list))       
        async def one(env_instance, m):
            chal = await env_instance.generate()
            result = await run(challenges=[chal], miners=[m], timeout=180)
            BUFFER[env_instance.key][m.key].extend(result)
            if len(BUFFER[env_instance.key][m.key]) >= K:
                logger.debug(f"Buffer full for {m.key[:18]}... in {env_instance.key}, sinking {K} results")
                await sink(wallet, env_instance, m, BUFFER[env_instance.key][m.key])
                BUFFER[env_instance.key][m.key] = []
                logger.debug(f"Buffer cleared for {m.key[:18]}... in {env_instance.key}")
            if not result[0].response.success:
                # We slash the weights here so that we are unlikely to query them again.
                nonlocal WEIGHTS
                WEIGHTS[m.key][env_instance.key] = WEIGHTS[m.key][env_instance.key]/2
            
        MAX_INFLIGHT = 30
        inflight: set[asyncio.Task] = set()    
        last_sync = 0           
        logger.info("Starting main runner loop")
        while True:
            try:
                
                # Finish current tasks.
                if len(inflight) >= MAX_INFLIGHT:
                    logger.debug(f"Max inflight reached ({MAX_INFLIGHT}), waiting for completion")
                    done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
                    for t in done: inflight.discard(t)
                    
                # Heartbeat and state update.
                global HEARTBEAT; HEARTBEAT = time.monotonic()                  
                if (time.time() - last_sync > 60*2) or (MINERS == None) or (WEIGHTS == None) or len(MINERS.keys()) == 0:
                    logger.info("Syncing state ...")
                    await sync_state()
                    last_sync = time.time()
                    continue
                
                # Select next env and miner.
                env_name, env_instance = random.choice(list(envs.items()))
                miner_objects = list(MINERS.values())
                weight_values = [WEIGHTS.get(m.key, {}).get(env_instance.key, 1.0) for m in miner_objects]
                # Fallback to uniform if all zeros
                if not any(w > 0 for w in weight_values):
                    weight_values = [1.0 for _ in miner_objects]
                miner = random.choices(miner_objects, weights=weight_values)[0]
                logger.debug(f"Selected miner: {miner.uid}, env: {env_instance.key} and weight: {WEIGHTS.get(miner.key, {}).get(env_instance.key, 0.0)} ")

                # Create and record the task
                t = asyncio.create_task(one(env_instance, miner))
                inflight.add(t)
                await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                logger.info("Runner cancelled, breaking loop")
                return
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Runner error: {type(e).__name__}: {str(e)}; retrying...")
                subtensor = await get_subtensor()
                await asyncio.sleep(5)
        
    async def main():
        logger.info("Starting main() with runner and watchdog")
        await asyncio.gather(_run(), watchdog(timeout=60*10))

    asyncio.run(main())


# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #
async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "",
) -> bool:
    for attempt in range(retries):
        try:
            st = await get_subtensor()
            ref = await st.get_current_block()
            logger.info(f"{log_prefix} set_weights attempt {attempt+1}/{retries}: netuid={netuid} uids={uids} weights={weights}")
            start = time.monotonic()
            bt.subtensor(get_conf('SUBTENSOR_ENDPOINT', default='finney')).set_weights(
                wallet=wallet, netuid=netuid, weights=weights, uids=uids,
                wait_for_inclusion=wait_for_inclusion,
            )
            logger.info(f"{log_prefix} extrinsic submitted in {(time.monotonic()-start)*1000:.1f}ms; waiting next block … (ref_block={ref})")
            await st.wait_for_block()
            meta = await st.metagraph(netuid)
            try:
                idx = meta.hotkeys.index(wallet.hotkey.ss58_address)
                lu = meta.last_update[idx]
                logger.info(f"{log_prefix} last_update={lu}, ref_block={ref}")
                if lu >= ref:
                    logger.info(f"{log_prefix} confirmation OK (last_update >= ref)")
                    return True
                logger.warning(f"{log_prefix} confirmation not yet included (last_update < ref), retrying …")
            except ValueError:
                logger.warning(f"{log_prefix} wallet hotkey not found in metagraph hotkeys; retrying …")
        except Exception as e:
            logger.warning(f"{log_prefix} set_weights attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}")
        await asyncio.sleep(delay_s)
    return False

@cli.command("signer")
@click.option('--host', default=os.getenv('SIGNER_HOST', '0.0.0.0'))
@click.option('--port', default=int(os.getenv('SIGNER_PORT', '8080')))
def signer(host: str, port: int):
    """Start lightweight HTTP signer service."""
    async def _run():
        from aiohttp import web
        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)
        @web.middleware
        async def access_log(request: "web.Request", handler):
            start = time.monotonic()
            try:
                resp = await handler(request)
                return resp
            finally:
                dur = (time.monotonic() - start) * 1000
                logger.info(
                    f"[signer] {request.remote} {request.method} {request.path} -> {getattr(request, 'response', None) and getattr(request.response, 'status', '?')} {dur:.1f}ms"
                )

        async def health(_request: "web.Request"):
            return web.json_response({"ok": True})
    
        async def sign_handler(request: "web.Request"):
            try:
                payload = await request.json()
                data = payload.get("payloads") or payload.get("data") or []
                if isinstance(data, str):
                    data = [data]
                sigs = [(wallet.hotkey.sign(data=d)).hex() for d in data]
                return web.json_response({
                    "success": True,
                    "signatures": sigs,
                    "hotkey": wallet.hotkey.ss58_address
                })
            except Exception as e:
                logger.error(f"[signer] /sign error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)


        async def set_weights_handler(request: "web.Request"):
            try:
                logger.info("[signer] /set_weights: request received")
                payload = await request.json()
                netuid = int(payload.get('netuid', NETUID))
                uids = payload.get('uids') or []
                weights = payload.get('weights') or []
                wait_for_inclusion = bool(payload.get('wait_for_inclusion', False))
                ok = await _set_weights_with_confirmation(
                    wallet,
                    netuid,
                    uids,
                    weights,
                    wait_for_inclusion,
                    retries=int(os.getenv("SIGNER_RETRIES", "10")),
                    delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
                    log_prefix="[signer]",
                )
                logger.info(f"[signer] /set_weights: confirmation={'ok' if ok else 'failed'}")
                return web.json_response({"success": True} if ok else {"success": False, "error": "confirmation failed"}, status=200 if ok else 500)
            except Exception as e:
                logger.error(f"[signer] set_weights error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        app = web.Application(middlewares=[access_log])
        app.add_routes([
            web.get('/healthz', health),
            web.post('/set_weights', set_weights_handler),
            web.post('/sign', sign_handler),
        ])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
        try:
            hn = socket.gethostname()
            ip = socket.gethostbyname(hn)
        except Exception:
            hn, ip = ("?", "?")
        logger.info(f"Signer service listening on http://{host}:{port} hostname={hn} ip={ip}")
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())

async def retry_set_weights( wallet: bt.Wallet, uids: List[int], weights: List[float], retry: int = 10 ):
    # Delegate to signer; fallback to shared helper only if signer is unreachable
    signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
    try:
        logger.info(f"Calling signer at {signer_url} for set_weights uids={uids}")
        parsed = urlparse(signer_url)
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
            addrs = ",".join(sorted({i[4][0] for i in infos}))
            logger.info(f"Signer DNS: host={parsed.hostname} -> {addrs}")
        except Exception as e:
            logger.warning(f"DNS resolve failed for {parsed.hostname}: {e}")
        timeout = aiohttp.ClientTimeout(connect=2, total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.monotonic()
            resp = await session.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": NETUID,
                    "weights": weights,
                    "uids": uids,
                    "wait_for_inclusion": False,
                },
            )
            dur_ms = (time.monotonic() - start) * 1000
            logger.info(f"Signer HTTP response status={resp.status} in {dur_ms:.1f}ms")
            # Try to parse JSON, otherwise log text (trimmed)
            try:
                data = await resp.json()
            except Exception:
                txt = await resp.text()
                data = {"raw": (txt[:500] + ('…' if len(txt) > 500 else ''))}
            logger.info(f"Signer response body={data}")
            if resp.status == 200 and data.get("success"):
                LASTSET.set(time.time())
                return
            # Do not fallback if signer exists but reports failure
            logger.warning(f"Signer responded error: status={resp.status} body={data}")
            return
    except ClientConnectorError as e:
        logger.info(f"Signer not reachable ({type(e).__name__}: {e}); falling back to local set_weights once")
        ok = await _set_weights_with_confirmation(
            wallet, NETUID, uids, weights, False,
            retries=int(os.getenv("SIGNER_RETRIES", "10")),
            delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
            log_prefix="[validator-fallback]",
        )
        if ok:
            LASTSET.set(time.time())
        else:
            logger.error("Local set_weights confirmation failed")
        return
    except asyncio.TimeoutError as e:
        logger.warning(f"Signer call timed out: {e}. Not falling back to local because validator has no wallet.")
        return
    
# --- Scoring hyperparameters --------------------------------------------------
TAIL = 20_000
ALPHA = 0.9

# Tuned ε-margins:
#  - 'not-worse' uses a smaller Z to ease dominance when sample sizes are large.
#  - 'better_any' uses a tiny fixed margin so small but consistent edges can win size-1 subsets.
EPS_FLOOR   = 0.002    # 0.20 percentage points floor for "not worse" tolerance
Z_NOT_WORSE = 0.84     # one-sided ~80% cushion for "not worse" (was 1.645)
EPS_WIN     = 0.0015   # 0.15 percentage points to claim "better on at least one env"
Z_WIN       = 0.0      # keep "better" threshold floor-based (set >0 to scale with n)
ELIG        = 0.6 

async def get_weights(tail: int = TAIL, scale: float = 1):
    """
    Compute miner weights using ε-Pareto dominance and combinatoric subset winners.

    Pipeline
      1) Ingest last `tail` blocks → per-miner per-env accuracy.
      2) Determine eligibility (>=90% of per-env max count).
      3) Global ε-dominance (all envs) for canonical 'best' (for tie breaks / summaries).
      4) Combinatoric scoring:
           - For every non-empty subset S of ENVS, pick the ε-Pareto winner on S.
           - Award K_|S| where K_1 = scale, K_s = C(N, s-1)*K_{s-1}.
         Fallback if no dominance edges on S: highest mean accuracy on S, then earliest version.
      5) Normalize scores over eligibles to produce weights. Metrics + summary emitted.

    Returns:
      (uids, weights): list of eligible UIDs (best last) and their weights (sum to 1).
    """

    # --- fetch + prune --------------------------------------------------------
    st = await get_subtensor()
    blk = await st.get_current_block()
    logger.info(f"Pruning {tail} blocks from {blk - tail} to {blk}")
    await prune(tail=tail)

    meta = await st.metagraph(NETUID)
    BASE_HK = meta.hotkeys[0]
    N_envs = len(ENVS)

    # Tallies for all known hotkeys (so metrics update is safe even if some have no data)
    cnt   = {hk: defaultdict(int)   for hk in meta.hotkeys}  # per-env counts
    succ  = {hk: defaultdict(int)   for hk in meta.hotkeys}  # per-env correct (0/1 or [0,1])
    prev  = {}                                                # last sample per hk
    v_id  = {}                                                # (model, revision) per hk
    first_block = {}                                          # earliest block for current version

    # --- ingest ---------------------------------------------------------------
    logger.info(f"Loading data from {blk - tail} to {blk}")
    async for c in dataset(tail=tail):
        NRESULTS.inc()
        hk, env = c.miner.hotkey, c.challenge.env.name

        # keep the base hk; otherwise require model family
        try:
            name = c.miner.model.split("/", 1)[1].lower()
        except Exception:
            name = str(c.miner.model).lower()
        if hk not in cnt or (hk != BASE_HK and not name.startswith("affine")):
            continue

        cur_vid = (c.miner.model, c.miner.revision)

        # On version change, reset ALL env streams and timestamp to current block
        if v_id.get(hk) != cur_vid:
            v_id[hk] = cur_vid
            first_block[hk] = c.miner.block
            for e in ENVS:
                cnt[hk][e] = 0
                succ[hk][e] = 0

        # accumulate on successes.
        prev[hk] = c
        if c.response.success:
            cnt[hk][env]  += 1
            succ[hk][env] += float(c.evaluation.score)

    logger.info("Collected results.")

    if not prev:
        logger.warning("No results collected; defaulting to uid 0")
        return 0, BASE_HK

    # --- accuracy + MAXENV ----------------------------------------------------
    acc = {
        hk: {e: (succ[hk][e] / cnt[hk][e] if cnt[hk][e] else 0.0) for e in ENVS}
        for hk in meta.hotkeys
    }

    active_hks = list(prev.keys())
    for e in ENVS:
        max_e = max((acc[hk][e] for hk in active_hks), default=0.0)
        MAXENV.labels(env=e).set(max_e)
    logger.info("Computed accuracy & updated MAXENV.")

    # --- eligibility: require near-max samples per env ------------------------
    async def _heads_for_active(active_hks: list[str]) -> dict[str, dict[str, int]]:
        by_env = {e: {} for e in ENVS}
        async def one(e: str, hk: str):
            try:
                by_env[e][hk] = await get_head(env=e, miner=hk)
            except Exception:
                by_env[e][hk] = 0
        await asyncio.gather(*(one(e, hk) for e in ENVS for hk in active_hks))
        return by_env

    heads_by_env = await _heads_for_active(active_hks)
    required = {e: int(ELIG * max(heads_by_env[e].values() or [0])) for e in ENVS}
    eligible = {
        hk for hk in active_hks
        if all(heads_by_env[e].get(hk, 0) >= required[e] for e in ENVS)
    }

    # --- ε-Pareto dominance helpers ------------------------------------------
    def thr_not_worse(a_i: float, n_i: int, a_j: float, n_j: int) -> float:
        """Tolerance for 'not worse' on an env: max(EPS_FLOOR, Z * SE_diff)."""
        if Z_NOT_WORSE <= 0:
            return EPS_FLOOR
        var = (a_i * (1 - a_i)) / max(n_i, 1) + (a_j * (1 - a_j)) / max(n_j, 1)
        return max(EPS_FLOOR, Z_NOT_WORSE * math.sqrt(var))

    def thr_better(a_i: float, n_i: int, a_j: float, n_j: int, nw: float) -> float:
        """
        Margin to claim 'better on at least one env'. Kept ≤ 'not worse' tolerance.
        Floor-based by default; set Z_WIN>0 to scale with SE_diff.
        """
        if Z_WIN > 0:
            var = (a_i * (1 - a_i)) / max(n_i, 1) + (a_j * (1 - a_j)) / max(n_j, 1)
            t = max(EPS_WIN, Z_WIN * math.sqrt(var))
        else:
            t = EPS_WIN
        return min(t, nw)

    def dominates_on(a: str, b: str, subset) -> bool:
        """
        True iff 'a' is not-worse than 'b' on every env in `subset` (within thr_not_worse),
        and strictly better on at least one env by thr_better. Full ε-ties break by earlier start.
        """
        not_worse_all = True
        better_any    = False
        tie_all       = True
        for e in subset:
            ai, aj = acc[a][e], acc[b][e]
            ni, nj = cnt[a][e], cnt[b][e]
            nw  = thr_not_worse(ai, ni, aj, nj)
            bet = thr_better(ai, ni, aj, nj, nw)

            if ai < aj - nw:
                not_worse_all = False
            if ai >= aj + bet:
                better_any = True
            if abs(ai - aj) > nw:
                tie_all = False

        if not_worse_all and better_any:
            return True
        if tie_all and first_block.get(a, float("inf")) < first_block.get(b, float("inf")):
            return True
        return False

    # Global dominance (full ENVS) for summary + canonical "best"
    dom_full = defaultdict(int)
    pool_for_dom = eligible if eligible else set(active_hks)
    for a, b in itertools.permutations(pool_for_dom, 2):
        if dominates_on(a, b, ENVS):
            dom_full[a] += 1
    logger.info("Computed ε-dominance counts (full env set).")

    def ts(hk: str) -> int:
        """Block-number timestamp; default to last seen block."""
        return int(first_block.get(hk, prev[hk].miner.block))

    best = max(pool_for_dom, key=lambda hk: (dom_full.get(hk, 0), -ts(hk))) if pool_for_dom else active_hks[0]
    best_uid = meta.hotkeys.index(best)

    # --- combinatoric scoring over all non-empty env subsets ------------------
    def layer_weights(N: int, kappa: float):
        """Per-subset weights K_s: K_1=kappa; K_s=C(N,s-1)*K_{s-1} for s>=2."""
        K = {1: kappa}
        for s in range(2, N + 1):
            K[s] = kappa * (2**s)
        return K

    def subset_winner(env_subset):
        """
        Winner on env_subset via ε-Pareto. If no dominance edges, fall back to:
          1) highest mean accuracy on the subset,
          2) earliest version start block.
        """
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool_for_dom, 2):
            if dominates_on(x, y, env_subset):
                dom_local[x] += 1

        def mean_acc(hk: str) -> float:
            return sum(acc[hk][e] for e in env_subset) / len(env_subset)

        return max(pool_for_dom, key=lambda hk: (dom_local.get(hk, 0), mean_acc(hk), -ts(hk)))

    # Calculate combinatoric scores for all miners (not just eligible)
    K = layer_weights(N_envs, scale)
    score = defaultdict(float)
    layer_points = {hk: defaultdict(float) for hk in active_hks}

    # --- Find single-env winners for highlighting ----------------------------
    env_winners = {}
    for e in ENVS:
        env_winners[e] = subset_winner((e,))

    # Award K_s to each subset winner
    for s in range(1, N_envs + 1):
        for env_subset in itertools.combinations(ENVS, s):
            w = subset_winner(env_subset)
            score[w] += K[s]
            layer_points[w][s] += K[s]

    # If no eligible miners exist, fall back to the canonical best with weight 1.0.
    if not eligible:
        logger.warning("No eligible miners; assigning weight 1.0 to canonical best.")
        for uid, hk in enumerate(meta.hotkeys):
            WEIGHT.labels(uid=uid).set(1.0 if hk == best else 0.0)
            for e in ENVS:
                a = acc[hk][e]
                if a > 0:
                    SCORE.labels(uid=uid, env=e).set(a)

        hdr = (
            ["UID", "Model", "Rev"]
            + [f"{e}" for e in ENVS]
            + [f"L{s}" for s in range(1, N_envs + 1)]
            + ["Pts", "Elig", "Wgt"]
        )
        def row(hk: str):
            m = prev[hk].miner
            w = 1.0 if hk == best else 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for e in ENVS:
                base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}"
                if hk == env_winners.get(e):
                    env_cols.append(f"*{base}*")
                else:
                    env_cols.append(base)
            return [
                m.uid, model_name, str(m.revision)[:5],
                *env_cols,
                *[f"{layer_points[hk][s]:.1f}" for s in range(1, N_envs + 1)],
                f"{score.get(hk, 0.0):.2f}",
                "Y" if hk in eligible else "N",
                f"{w:.4f}",
            ]
        rows = sorted((row(hk) for hk in active_hks), key=lambda r: (r[-3], r[0]), reverse=True)
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))
        return [best_uid], [1.0]

    # Eligible path: normalize scores to weights over the eligible pool only
    total_points = sum(score[hk] for hk in eligible)
    if total_points <= 0:
        logger.warning("Combinatoric scoring returned zero total; falling back to canonical best.")
        weight_by_hk = {hk: (1.0 if hk == best else 0.0) for hk in eligible}
    else:
        weight_by_hk = {hk: (score[hk] / total_points) for hk in eligible}

    # --- summary printout -----------------------------------------------------
    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in ENVS]
        + [f"L{s}" for s in range(1, N_envs + 1)]
        + ["Pts", "Elig", "Wgt"]
    )
    def row(hk: str):
        m = prev[hk].miner
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for e in ENVS:
            base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}"
            if hk == env_winners.get(e):
                env_cols.append(f"*{base}*")
            else:
                env_cols.append(base)
        return [
            m.uid, model_name[:30], str(m.revision)[:5],
            *env_cols,
            *[f"{layer_points[hk][s]:.1f}" for s in range(1, N_envs + 1)],
            f"{score.get(hk, 0.0):.2f}",
            "Y" if hk in eligible else "N",
            f"{w:.4f}",
        ]
    ranked_rows   = sorted((row(hk) for hk in eligible), key=lambda r: float(r[-3]), reverse=True)
    unranked_rows = sorted((row(hk) for hk in active_hks if hk not in eligible), key=lambda r: float(r[-3]), reverse=True)
    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    # --- Prometheus updates ---------------------------------------------------
    for uid, hk in enumerate(meta.hotkeys):
        WEIGHT.labels(uid=uid).set(weight_by_hk.get(hk, 0.0))
        for e in ENVS:
            a = acc[hk][e]
            if a > 0:
                SCORE.labels(uid=uid, env=e).set(a)

    # --- Return weights in a stable shape (best last, as before) -------------
    eligible_uids = [meta.hotkeys.index(hk) for hk in eligible]
    uids = [u for u in eligible_uids if u != best_uid] + [best_uid]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights


        
@cli.command("validate")
def validate():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)    
    async def _run():     
        LAST = 0
        TEMPO = 100
        subtensor = None
        while True:
            try:
                # ---------------- Wait for set weights. -----------------
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
                if subtensor is None: subtensor = await get_subtensor()
                BLOCK = await subtensor.get_current_block()
                if BLOCK % TEMPO != 0 or BLOCK <= LAST: 
                    logger.debug(f'Waiting ... {BLOCK} % {TEMPO} == {BLOCK % TEMPO} != 0')
                    await subtensor.wait_for_block()
                    continue
                
                # ---------------- Set weights. ------------------------
                uids, weights = await get_weights()
        
                # ---------------- Set weights. ------------------------
                logger.info("Setting weights ...")
                await retry_set_weights( wallet, uids=uids, weights=weights, retry = 3)
                subtensor = await get_subtensor()
                SETBLOCK = await subtensor.get_current_block()
                LASTSET.set_function(lambda: SETBLOCK - LAST)
                LAST = BLOCK           
            
                # ---------------- Other telemetry ------------------------
                CACHE.set(sum( f.stat().st_size for f in CACHE_DIR.rglob("*.json") if f.is_file()))
                
            except asyncio.CancelledError: break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
            
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 20))
        )
    asyncio.run(main())
    
    
@cli.command("weights")
def weights():
    asyncio.run(get_weights())

# --------------------------------------------------------------------------- #
#                              Pull Model                                     #
# --------------------------------------------------------------------------- #
@cli.command("pull")
@click.argument("uid", type=int)
@click.option("--model_path", "-p", default = './model_path', required=True, type=click.Path(), help="Local directory to save the model")
@click.option('--hf-token', default=None, help="Hugging Face API token (env HF_TOKEN if unset)")
def pull(uid: int, model_path: str, hf_token: str):
    """Pulls a model from a specific miner UID if exists."""

    # 1. Ensure HF token
    hf_token     = hf_token or get_conf("HF_TOKEN")

    # 2. Lookup miner on‑chain
    miner_map = asyncio.run(miners(uids=uid))
    miner = miner_map.get(uid)
    
    if miner is None:
        click.echo(f"No miner found for UID {uid}", err=True)
        sys.exit(1)
    repo_name = miner.model
    logger.info("Pulling model %s for UID %d into %s", repo_name, uid, model_path)

    # 3. Download snapshot
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=miner.revision,
        )
        click.echo(f"Model {repo_name} pulled to {model_path}")
    except Exception as e:
        logger.error("Failed to download %s: %s", repo_name, e)
        click.echo(f"Error pulling model: {e}", err=True)
        sys.exit(1)


# --------------------------------------------------------------------------- #
#                              Push Model                                     #
# --------------------------------------------------------------------------- #
@cli.command("push")
@click.option('--model_path',  default='./model_path', help='Local path to model artifacts.')
@click.option('--existing-repo', default=None, help='Use an existing HF repo instead of uploading (format <user>/<repo>)')
@click.option('--revision', default=None, help='Commit SHA to register (only relevant with --existing-repo)')
@click.option('--coldkey',     default=None, help='Name of the cold wallet to use.')
@click.option('--hotkey',      default=None, help='Name of the hot wallet to use.')
@click.option('--chutes-api-key', default=None, help='Chutes API key (env CHUTES_API_KEY if unset)')
def push(model_path: str, existing_repo: str, revision: str, coldkey: str, hotkey: str, chutes_api_key: str):
    """Pushes a model to be hosted by your miner."""
    # -----------------------------------------------------------------------------
    # 1. Wallet & config
    # -----------------------------------------------------------------------------
    coldkey = coldkey or get_conf("BT_WALLET_COLD", "default")
    hotkey  = hotkey  or get_conf("BT_WALLET_HOT", "default")
    logger.debug("Using coldkey=%s, hotkey=%s", coldkey, hotkey)
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    # Required API credentials
    hf_user        = get_conf("HF_USER")
    hf_token       = get_conf("HF_TOKEN")
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user     = get_conf("CHUTE_USER")
    # TODO: validate API creds, exit gracefully if missing

    # -----------------------------------------------------------------------------
    # 2. Prepare HF repo name - If --existing-repo provided, use it directly and skip local upload
    # -----------------------------------------------------------------------------
    repo_name = existing_repo or f"{hf_user}/Affine-{wallet.hotkey.ss58_address}"
    logger.debug("Using existing HF repo: %s" if existing_repo else "Hugging Face repo: %s", repo_name)

    # -----------------------------------------------------------------------------
    # 3. Create & secure HF repo
    # -----------------------------------------------------------------------------
    api = HfApi(token=hf_token)
    if not existing_repo:
        api.create_repo(repo_id=repo_name, repo_type="model", private=True, exist_ok=True)
        try: api.update_repo_visibility(repo_id=repo_name, private=True)
        except Exception: logger.debug("Repo already private or visibility update failed")

    # -----------------------------------------------------------------------------
    # 4. Upload model files to HF (skip if using existing repo)
    # -----------------------------------------------------------------------------
    async def deploy_model_to_hf():
        logger.debug("Starting model upload from %s", model_path)
        # Gather files
        files = []
        for root, _, fnames in os.walk(model_path):
            if ".cache" in root or any(p.startswith(".") for p in root.split(os.sep)):
                continue
            for fname in fnames:
                if not (fname.startswith(".") or fname.endswith(".lock")):
                    files.append(os.path.join(root, fname))

        # Upload files with limited concurrency to avoid HF 429 errors
        SEM = asyncio.Semaphore(int(os.getenv("AFFINE_UPLOAD_CONCURRENCY", "2")))

        async def _upload(path: str):
            rel = os.path.relpath(path, model_path)
            async with SEM:  # limit concurrent commits
                await asyncio.to_thread(
                    lambda: api.upload_file(
                        path_or_fileobj=path,
                        path_in_repo=rel,
                        repo_id=repo_name,
                        repo_type="model"
                    )
                )
                logger.debug("Uploaded %s", rel)

        await asyncio.gather(*(_upload(p) for p in files))
        logger.debug("Model upload complete (%d files)", len(files))

    asyncio.run(deploy_model_to_hf()) if not existing_repo else logger.debug("Skipping model upload because --existing-repo was provided")

    # -----------------------------------------------------------------------------
    # 5. Fetch latest revision hash
    # -----------------------------------------------------------------------------
    if revision:
        logger.debug("Using user-supplied revision: %s", revision)
    else:
        info      = api.repo_info(repo_id=repo_name, repo_type="model")
        revision  = getattr(info, "sha", getattr(info, "oid", "")) or ""
        logger.debug("Latest revision from HF: %s", revision)

    # -----------------------------------------------------------------------------
    # 6. Commit model revision on-chain
    # -----------------------------------------------------------------------------
    chute_id = None

    async def commit_to_chain():
        """Submit the model commitment, retrying on quota errors."""
        logger.debug("Preparing on-chain commitment")
        sub     = await get_subtensor()
        payload = json.dumps({"model": repo_name, "revision": revision, "chute_id": chute_id})
        while True:
            try:
                await sub.set_reveal_commitment(wallet=wallet, netuid=NETUID, data=payload, blocks_until_reveal=1)
                logger.debug("On-chain commitment submitted")
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    logger.debug("SpaceLimitExceeded – waiting one block before retrying")
                    await sub.wait_for_block()
                else:
                    raise


    # -----------------------------------------------------------------------------
    # 7. Make HF repo public
    # -----------------------------------------------------------------------------
    try:
        api.update_repo_visibility(repo_id=repo_name, private=False)
        logger.debug("Repo made public")
    except Exception:
        logger.trace("Failed to make repo public (already public?)")

    # -----------------------------------------------------------------------------
    # 8. Deploy Chute
    # -----------------------------------------------------------------------------
    async def deploy_to_chutes():
        logger.debug("Building Chute config")
        rev_flag = f'revision="{revision}",' if revision else ""
        chutes_config = textwrap.dedent(f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_name}",
    model_name="{repo_name}",
    image="chutes/sglang:0.4.9.post3",
    concurrency=20,
    {rev_flag}
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=24,
    ),
    engine_args=(
        "--trust-remote-code "
    ),
)
""")
        tmp_file = Path("tmp_chute.py")
        tmp_file.write_text(chutes_config)
        logger.debug("Wrote Chute config to %s", tmp_file)
        logger.debug("=== chute file ===\n%s", tmp_file.read_text())

        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--public"]
        env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        # Auto-answer the interactive Y/N prompt
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        stdout, _ = await proc.communicate()
        output = stdout.decode().split('confirm? (y/n)')[1].strip()
        logger.trace(output)

        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)', output)
        if match and match.group(2) == "ERROR":
            logger.debug("Chutes deploy failed with the above error log")
            raise RuntimeError("Chutes deploy failed")
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    # -----------------------------------------------------------------------------
    # 8b. Retrieve chute_id and commit on-chain
    # -----------------------------------------------------------------------------
    chute_id = asyncio.run(get_latest_chute_id(repo_name, api_key=chutes_api_key))

    asyncio.run(commit_to_chain())

    # -----------------------------------------------------------------------------
    # 9. Warm up model until it’s marked hot
    # -----------------------------------------------------------------------------
    async def warmup_model():
        logger.debug("Warming up model with SAT challenges")
        sub       = await get_subtensor()
        meta      = await sub.metagraph(NETUID)
        my_uid    = meta.hotkeys.index(wallet.hotkey.ss58_address)
        miner  = (await miners(netuid=NETUID))[my_uid]

        while not (miner.chute or {}).get("hot", False):
            from .envs import SAT
            challenge = await SAT().generate()
            await run(challenges=challenge, miners=[miner])
            await sub.wait_for_block()
            miner = (await miners(netuid=NETUID))[my_uid]
            logger.trace("Checked hot status: %s", (miner.chute or {}).get("hot"))

        logger.debug("Model is now hot and ready")

    asyncio.run(warmup_model())
    logger.debug("Mining setup complete. Model is live!")  
