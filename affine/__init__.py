#!/usr/bin/env python3
import os
import io
import sys
import json
import time
import math
import types 
import click
import random
import aiohttp
import asyncio
import logging
import textwrap
import botocore
import numpy as np
import bittensor as bt
import botocore.config
from rich.table import Table
from dotenv import load_dotenv
from rich.console import Console
from collections import defaultdict
from abc import ABC, abstractmethod
from alive_progress import alive_bar
from pydantic import BaseModel, Field
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence

NETUID = 120

# ── LOGGING ─────────────────────────────────────────────────────────────
TRACE = 5
logging.addLevelName(TRACE, "TRACE")
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    if verbosity >= 3: level = TRACE
    elif verbosity == 2: level = logging.DEBUG
    elif verbosity == 1: level = logging.INFO
    else: level = logging.CRITICAL + 1
    for noisy_logger in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
# ── SUBTENSOR ─────────────────────────────────────────────────────────────
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.debug("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor()
        await SUBTENSOR.initialize()
        logger.debug("Connected")
    return SUBTENSOR

# ── CLI ─────────────────────────────────────────────────────────────────
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)
    
# ── Config. ─────────────────────────────────────────────────────────────────
load_dotenv(os.path.expanduser("~/.affine/config.env"), override=True)
load_dotenv(override=True)
def get_conf(key) -> Any:
    value = os.getenv(key)
    if not value:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True)
        sys.exit(1)
    return value

@cli.command('set')
@click.argument('key')
@click.argument('value')
def set_config(key: str, value: str):
    """Set a key-value pair in ~/.affine/config.env."""
    path = os.path.expanduser("~/.affine/config.env")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [l for l in open(path).readlines() if not l.startswith(f"{key}=")] if os.path.exists(path) else []
    lines.append(f"{key}={value}\n")
    open(path, "w").writelines(lines)
    logger.info("Set %s in %s", key, path)
    click.echo(f"Set {key} in {path}")
    
# ── Models ───────────────────────────────────────────────────────────────
def _truncate(text: Optional[str], max_length: int = 80) -> str:
    if not text:
        return ""
    return textwrap.shorten(text, width=max_length, placeholder="...")

class Miner(BaseModel):
    uid: int
    hotkey: str
    model: Optional[str] = None
    revision: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]
    success: bool
    def __repr__(self) -> str:
        resp = _truncate(self.response)
        err  = _truncate(self.error)
        return (
            f"<Response model={self.model!r} success={self.success} "
            f"latency={self.latency_seconds:.3f}s attempts={self.attempts} "
            f"response={resp!r} error={err!r}>"
        )
    __str__ = __repr__

class BaseEnv(BaseModel, ABC):
    @property
    def name(self) -> str:return self.__class__.__name__
    class Config: arbitrary_types_allowed = True
    def __hash__(self): return hash(self.name)
    def __repr__(self): return self.name
    async def many(self, n: int) -> List["Challenge"]:
        return await asyncio.gather(*(self.generate() for _ in range(n)))
    @abstractmethod
    async def generate(self) -> "Challenge": ...
    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: Response) -> "Evaluation": ...

class Challenge(BaseModel):
    env: BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    async def evaluate(self, response: Response) -> "Evaluation":
        return await self.env.evaluate(self, response)
    def __repr__(self) -> str:
        pr = _truncate(self.prompt)
        return f"<Challenge env={self.env.name!r} prompt={pr!r}>"
    __str__ = __repr__

class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)
    def __repr__(self) -> str:
        ex = {k: _truncate(str(v)) for k, v in self.extra.items()}
        return f"<Evaluation env={self.env.name!r} score={self.score:.4f} extra={ex!r}>"
    __str__ = __repr__

class Result(BaseModel):
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation
    
    
# ── S3 ─────────────────────────────────────────────────────────────────
BUCKET     = get_conf("R2_BUCKET_ID")
ACCOUNT    = get_conf("R2_ACCOUNT_ID")
KEY_ID     = get_conf("R2_WRITE_ACCESS_KEY_ID")
SECRET     = get_conf("R2_WRITE_SECRET_ACCESS_KEY")
ENDPOINT   = f"https://{ACCOUNT}.r2.cloudflarestorage.com"
CLIENT_CFG = botocore.config.Config(max_pool_connections=256)
CLIENT     = None

async def get_client():
    global CLIENT
    if CLIENT is None:
        session = get_session()
        CLIENT = await session.create_client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET,
            config=CLIENT_CFG,
        ).__aenter__()
    return CLIENT

async def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, default=lambda o: o.json() if hasattr(o, "json") else o).encode()

async def get(key: str) -> Any:
    client = await get_client()
    response = await client.get_object(Bucket=BUCKET, Key=key)
    body = await response["Body"].read()
    content_type = response.get("ContentType", "")
    if content_type == "application/json": return json.loads(body.decode("utf-8"))
    else: return body

async def sink(key: str, obj: Any, *, content_type: str = "application/json"):
    client = await get_client()
    body = await _json_bytes(obj) if content_type == "application/json" else obj
    await client.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=content_type)

# ── API ───────────────────────────────────────────────────────────────
async def get_chute(model: str) -> Dict[str, Any]:
    url = f"https://api.chutes.ai/chutes/{model}"
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
            logger.trace("Fetched chute info for %s", model)
            return info
        
async def get_chute_code(identifier: str) -> Optional[str]:
    """Return raw chute Python code for *identifier* or ``None``.

    The endpoint is `https://api.chutes.ai/chutes/code/{identifier}`.
    """
    url = f"https://api.chutes.ai/chutes/code/{identifier}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            return await r.text(errors="ignore")

def _extract_revision(code: str) -> Optional[str]:
    """Extract `--revision <sha>` from chute code string."""
    import re
    m = re.search(r"--revision\s+([0-9a-f]{40})", code)
    return m.group(1) if m else None

async def miners(
        uids: Optional[Union[int, List[int]]] = None, 
        no_null: bool = False, 
        netuid: int = NETUID,
        min_block: int = 0,
        metagraph: object = None,
    ) -> Dict[int, Miner]:
    subtensor = await get_subtensor()
    if metagraph is None:
        metagraph = await subtensor.metagraph(netuid)
    revs = await subtensor.get_all_revealed_commitments(netuid)
    if uids is None: uids = list(range(len(metagraph.hotkeys)))
    elif isinstance(uids, int): uids = [uids]
    async def _get_miner(uid: int) -> Optional[Miner]:
        if not (0 <= uid < len(metagraph.hotkeys)): return None
        hotkey = metagraph.hotkeys[uid]
        commits = revs.get(hotkey) or []
        if not commits: return None # Filter on null.
        block, model_data = commits[-1]
        revision = None
        # Nouveau format JSON {"model":..., "revision":...}
        if isinstance(model_data, str) and model_data.strip().startswith("{"):
            try:
                parsed = json.loads(model_data)
                model_data = parsed.get("model")
                revision = parsed.get("revision")
            except Exception:
                pass
        model = model_data
        if no_null and block is None: return None # Filter on null.
        if no_null and model is None: return None # Filter on null.
        if block < min_block: return None # Filter on block.
        chute = await get_chute(str(model))
        if chute is None:
            # fall back to code endpoint using model as identifier
            code = await get_chute_code(str(model))
            revision = _extract_revision(code or "") if code else None
        else:
            # Try extract revision from returned info first
            if 'revision' in chute and chute['revision']:
                revision = str(chute['revision'])
            else:
                # fallback: fetch code via uuid if available
                cid = chute.get('id') or chute.get('uuid')
                if cid:
                    code = await get_chute_code(cid)
                    revision = _extract_revision(code or "") if code else None
        if no_null and chute is None: return None # Filter on no chute.
        miner = Miner(uid=uid, hotkey=hotkey, model=str(model), revision=revision, block=int(block), chute=chute)
        return miner
    miners = {uid: miner for uid in uids if (miner := await _get_miner(uid)) is not None}
    return miners

# ── run challenges ────────────────────────────────────────────────────────────
HTTP_SEM = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
TERMINAL = {400, 404, 410}

async def run(challenges, miners, timeout=120, retries=0, backoff=1, progress=True):
    if not isinstance(challenges, list): challenges = [challenges]
    if isinstance(miners, dict):  mmap = miners
    elif isinstance(miners, list) and all(hasattr(m, "uid") for m in miners):  mmap = {m.uid: m for m in miners}
    else: mmap = await miners(miners)
    logger.trace("Running challenges: %s on miners: %s", [chal.prompt[:30] for chal in challenges], list(mmap.keys()))

    async def _run_one(chal, model):
        url = "https://llm.chutes.ai/v1/chat/completions"
        hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
        start = time.monotonic()
        logger.trace("Starting %r on model=%s", chal.prompt[:30], model)
        for attempt in range(1, retries + 2):
            try:
                async with HTTP_SEM, sess.post(url, json={"model": model, "messages":[{"role":"user","content":chal.prompt}]}, headers=hdr, timeout=timeout) as r:
                    text = await r.text(errors="ignore")
                    logger.trace("HTTP %d on attempt %d for %s", r.status, attempt, model)
                    if r.status in TERMINAL:
                        err = f"{r.status}:{text}"
                        logger.debug("Terminal error for %s: %s", model, err)
                        return Response(response=None, latency_seconds=time.monotonic()-start, attempts=attempt, model=model, error=err, success=False)
                    if r.status != 200:
                        raise RuntimeError(f"{r.status}:{text}")
                    body = await r.json()
                    res = body["choices"][0]["message"]["content"]
                    logger.trace("Success for %s in %.2fs (attempt %d)", model, time.monotonic()-start, attempt)
                    return Response(response=res, latency_seconds=time.monotonic()-start, attempts=attempt, model=model, error=None, success=True)
            except Exception as e:
                logger.debug("Error on attempt %d for %s: %s", attempt, model, e)
                if attempt > retries:
                    return Response(response=None, latency_seconds=time.monotonic()-start, attempts=attempt, model=model, error=str(e), success=False)
                await asyncio.sleep(backoff * 2**(attempt-1) * (1 + random.uniform(-.1, .1)))

    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        async def proc(m, chal):
            resp = await _run_one(chal, m.model)
            ev = await chal.evaluate(resp)
            logger.trace("Evaluation done for %s: %r", m.model, ev)
            return Result(miner=m, challenge=chal, response=resp, evaluation=ev)

        tasks = [asyncio.create_task(proc(m, chal))
                 for m in mmap.values() if m.model for chal in challenges]
        if progress:
            with alive_bar(len(tasks), title="Running challenges") as bar:
                for task in asyncio.as_completed(tasks):
                    results.append(await task); bar()
        else:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
    logger.trace("Finished all runs (%d results)", len(results))
    return results


# ── CLI & commands ────────────────────────────────────────────────────────────
# Import environments
from .envs.sat import SAT
from .envs.abduction import ABDUCTION
from .envs.math import MATH

# Registry of active environments
ENVS = {"SAT": SAT, "ABDUCTION": ABDUCTION, "MATH": MATH}

@cli.command("validate")
def validate():
    console = Console()
    wallet = bt.wallet()

    async def main():
        K, alpha = 10, 0.999
        subtensor = await get_subtensor()
        meta      = await subtensor.metagraph(NETUID)

        # — Initial state
        miners_map   = await miners(no_null=True, metagraph=meta)
        hotkeys      = [m.hotkey for m in miners_map.values()]
        blocks       = {m.hotkey: m.block for m in miners_map.values()}
        scores       = {m.hotkey: defaultdict(float) for m in miners_map.values()}
        window_start = await subtensor.get_current_block()

        while True:
            # — Refresh miners
            meta       = await subtensor.metagraph(NETUID)
            miners_map = await miners(no_null=True, metagraph=meta)
            miners_by_hotkey = {m.hotkey: m for m in miners_map.values()}
            for m in miners_map.values():
                if m.hotkey not in blocks:
                    hotkeys.append(m.hotkey)
                    blocks[m.hotkey] = m.block
                    scores[m.hotkey] = defaultdict(float)
                elif blocks[m.hotkey] != m.block:
                    blocks[m.hotkey] = m.block
                    scores[m.hotkey] = defaultdict(float)

            # — Collect for K blocks
            while await subtensor.get_current_block() < window_start + K:
                blk = await subtensor.get_current_block()
                challenges = [await env().generate() for env in ENVS.values()]
                results    = await run(challenges=challenges, miners=miners_map)
                await sink(f"affine/results/{wallet.hotkey.ss58_address}/{blk:08d}.json",
                           [r.json() for r in results])
                for r in results:
                    e, hk, raw = r.challenge.env.name, r.miner.hotkey, r.evaluation.score
                    prev = scores[hk][e]
                    scores[hk][e] = raw * (1 - alpha) + prev * alpha

            # — Compute dense ranks & custom dominance counts
            ranks, counts = {}, defaultdict(int)
            for e in ENVS:
                uniq = sorted({scores[h][e] for h in hotkeys}, reverse=True)
                rank_of = {score: i+1 for i, score in enumerate(uniq)}
                ranks[e] = {h: rank_of[scores[h][e]] for h in hotkeys}

            env_list = list(ENVS)
            for a in hotkeys:
                for b in hotkeys:
                    if a == b:
                        continue
                    # new rule:
                    # a "beats" b if
                    #   - a is never worse than b on any env (<=)
                    #   - a is strictly better on more than one env
                    better_count = sum(ranks[e][a] < ranks[e][b] for e in env_list)
                    not_worse    = all(ranks[e][a] <= ranks[e][b] for e in env_list)
                    if not_worse and better_count >= 1:
                        counts[a] += 1

            # — Pick best (most custom wins), tie‑break by oldest block
            best_key, best = (-1, None), None
            for h in hotkeys:
                key = (counts.get(h, 0), -blocks.get(h, 0))
                if key > best_key:
                    best_key, best = key, h

            # — Prepare weights
            weights = [1.0 if hk == best else 0.0 for hk in meta.hotkeys]

            # — Sink snapshot
            snap_key = f"affine/snapshots/{wallet.hotkey.ss58_address}/{window_start:08d}.json"
            await sink(snap_key, {
                "window_start": window_start,
                "scores":       {hk: dict(scores[hk]) for hk in hotkeys},
                "blocks":       blocks,
                "weights":      weights,
                "miners":       {hk: {'uid': miner.uid, 'model': miner.model} for hk, miner in miners_map.items()}
            })

            # — Render Rich table
            table = Table(title=f"Window {window_start}–{window_start+K}")
            table.add_column("Miner UID", style="bold")
            table.add_column("Block", justify="right")
            table.add_column("Model", justify="right")
            for e in ENVS:
                table.add_column(f"{e} Score", justify="right")
                table.add_column(f"{e} Rank", justify="right")
            table.add_column("Weight", justify="right")

            for hk in hotkeys:
                if hk not in miners_by_hotkey:
                    continue
                miner = miners_by_hotkey[hk]
                row = [str(miner.uid), str(miner.block), miner.model]
                for e in ENVS:
                    row.extend([f"{scores[hk][e]:.4f}", str(ranks[e][hk])])
                row.append(f"{weights[meta.hotkeys.index(hk)]:.2f}")
                table.add_row(*row)
            console.print(table)

            # — Submit weights on chain
            await subtensor.set_weights(
                wallet=wallet,
                netuid=NETUID,
                uids=meta.uids,
                weights=weights,
                wait_for_inclusion=False
            )

            # — Slide window
            window_start += K

    asyncio.run(main())

