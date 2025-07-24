
#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import os
import re
import sys
import json
import time
import click
import random
import aiohttp
import asyncio
import logging
import textwrap
import traceback
from .utils import *
import datetime as dt
import bittensor as bt
from pathlib import Path
from tabulate import tabulate
from dotenv import load_dotenv
from huggingface_hub import HfApi
from botocore.config import Config
from collections import defaultdict
from abc import ABC, abstractmethod
from aiobotocore.session import get_session
from huggingface_hub import snapshot_download
from bittensor.core.errors import MetadataError
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
# Added deque for bounded result cache
from typing import List, Dict, Any, Union, Optional, Deque
from collections import deque


__version__ = "0.0.0"

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
NETUID = 120
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")

# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #
LOCAL_BASE = Path.home() / ".affine"; LOCAL_BASE.mkdir(parents=True, exist_ok=True)
load_dotenv(os.path.expanduser("~/.affine/config.env"), override=True); load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key); 
    if not v and default is None:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True); sys.exit(1)
    return v or default

# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.trace("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor()
        await SUBTENSOR.initialize()
        logger.trace("Connected")
    return SUBTENSOR

# --------------------------------------------------------------------------- #
#                           Base‑level data models                            #
# --------------------------------------------------------------------------- #
def _truncate(t: Optional[str], max_len: int = 80) -> str:
    return "" if not t else textwrap.shorten(t, width=max_len, placeholder="…")

class BaseEnv(BaseModel, ABC):
    """Abstract competition environment."""
    class Config: arbitrary_types_allowed = True
    @property
    def name(self) -> str: return self.__class__.__name__
    def __hash__(self):     return hash(self.name)
    def __repr__(self):     return self.name
    # API expected from concrete envs
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
    @validator("env", pre=True)
    def _parse_env(cls, v):
        from .envs.sat import SAT
        from .envs.abd import ABD
        from .envs.ded import DED
        ENVS = {"SAT": SAT, "ABD": ABD, "DED": DED}
        return ENVS[v]() if isinstance(v, str) else v
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
        from .envs.sat import SAT
        from .envs.abd import ABD
        from .envs.ded import DED
        ENVS = {"SAT": SAT, "ABD": ABD, "DED": DED}
        return ENVS[v]() if isinstance(v, str) else v
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

# Real import.    
from .envs.sat import SAT
from .envs.abd import ABD
from .envs.ded import DED
ENVS = {"SAT": SAT, "ABD": ABD, "DED": DED}

# --------------------------------------------------------------------------- #
#                   S3 helpers                                                #
# --------------------------------------------------------------------------- #
get_client_ctx = lambda: get_session().create_client(
    "s3",
    endpoint_url=f"https://{get_conf('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=get_conf("R2_WRITE_ACCESS_KEY_ID"),
    aws_secret_access_key=get_conf("R2_WRITE_SECRET_ACCESS_KEY"),
    config=Config(max_pool_connections=256)
)

async def sink(wallet: bt.wallet, block: int, results: list[Result]):
    key = f"affine/results/{wallet.hotkey.ss58_address}/{block}.json"
    body = json.dumps([r.sign(wallet) or r.model_dump(mode="json") for r in results]).encode()
    logger.debug(f"[SINK] block={block} → key={key}")
    try:
        async with get_client_ctx() as client:
            await client.put_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key, Body=body)
    except Exception: logger.error("R2 write failed for %s", key, exc_info=True)

async def get(key: str):
    async with get_client_ctx() as client:
        resp = await client.get_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
        data = await resp["Body"].read()
    return json.loads(data) if resp.get("ContentType") == "application/json" else data

async def dataset(tail: int = 1000, max_concurrency: int = 10) -> list[Result]:
    sub = await get_subtensor()
    cutoff = (await sub.get_current_block()) - tail
    bucket, prefix = get_conf("R2_BUCKET_ID"), "affine/results/"
    # collect all keys newer than cutoff
    keys = []
    async with get_client_ctx() as s3:
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        async for page in pages:
            for o in page.get("Contents", []):
                k = o["Key"]
                try:
                    if int(k.rsplit("/", 1)[-1].removesuffix(".json")) > cutoff: keys.append(k)
                except ValueError: pass

    sem = asyncio.Semaphore(max_concurrency)
    async def fetch(k):
        async with sem:
            d = await get(k)
            return json.loads(d) if isinstance(d, (bytes, bytearray)) else d

    results = []
    # fetch & rehydrate
    batches = await asyncio.gather(*[fetch(k) for k in sorted(keys)])
    for batch in batches:
        if isinstance(batch, list):
            for item in batch:
                try:
                    r = Result.model_validate(item)
                    if r.verify(): results.append(r)
                except Exception: logger.trace("Deserialization error", exc_info=True)
    return results


# --------------------------------------------------------------------------- #
#                               QUERY                                         #
# --------------------------------------------------------------------------- #
HTTP_SEM = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
TERMINAL = {400, 404, 410}

async def query(prompt, model="unsloth/gemma-3-12b-it", timeout=120, retries=0, backoff=1) -> Response:
    url = "https://llm.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
    start = time.monotonic()
    R = lambda resp, at, err, ok: Response(response=resp, latency_seconds=time.monotonic()-start,
                                          attempts=at, model=model, error=err, success=ok)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        for attempt in range(1, retries+2):
            try:
                async with HTTP_SEM, sess.post(url, json={"model": model, "messages":[{"role":"user","content":prompt}]},
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
async def run(challenges, miners, timeout=120, retries=0, backoff=1, progress=True) -> List[Result]:
    if not isinstance(challenges, list): challenges = [challenges]
    if isinstance(miners, Miner): miners = [miners]
    if isinstance(miners, dict):  mmap = miners
    elif isinstance(miners, list) and all(hasattr(m, "uid") for m in miners):  mmap = {m.uid: m for m in miners}
    else: mmap = await miners(miners)
    logger.trace("Running challenges: %s on miners: %s", [chal.prompt[:30] for chal in challenges], list(mmap.keys()))
    response = []
    async def proc(miner, chal):
        resp = await query(chal.prompt, miner.model, timeout, retries, backoff)
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

def _extract_revision(code: str) -> Optional[str]:
    matches = re.findall(r"--revision\s+([0-9a-f]{40})", code)
    # only accept exactly one occurrence
    if len(matches) == 1: return matches[0]
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
            data = json.loads(data)
            model, miner_revision = data.get("model"), data.get("revision")
            chute = await get_chute(model)
            code = await get_chute_code(chute['chute_id'])
            chutes_revision = _extract_revision(code)
            if chutes_revision == None or miner_revision == chutes_revision:
                miner = Miner(
                    uid=uid, hotkey=hotkey, model=model, block=int(block),
                    revision = miner_revision,
                    chute=chute,
                )
                return miner
        except: pass
    results = await asyncio.gather(*(fetch(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}
    return output
        

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)

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

@cli.command("validate")
@click.option("--coldkey", default=None, help="Cold wallet name")
@click.option("--hotkey", default=None, help="Hot wallet key")
def validate(coldkey: str, hotkey: str):
    
    # ---------------- Init Wallet ------------------------
    coldkey = coldkey or get_conf("BT_WALLET_COLD", "default")
    hotkey  = hotkey  or get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # ---------------- Set weights. ------------------------
    LAST = 0
    ALPHA = 0.9
    TEMPO = 100
    # Use bounded deque to prevent unbounded memory growth
    RESULTS: Deque[Result] = deque(maxlen=10000)  # keep most-recent 10k results (~ a few MB)
    async def maybe_set_weights( sub, meta, block ): 
        nonlocal LAST, RESULTS       
        if block - LAST < TEMPO: return
        # Pull and extend RESULTS with all results from validators.
        next_results = await dataset( tail = min(block - LAST, 1000) )
        for res in next_results:
            if res.hotkey in meta.hotkeys and bool(meta.validator_permit[ meta.hotkeys.index( res.hotkey) ]):
                RESULTS.append( res )  # leftmost items are automatically discarded when over limit
        LAST = block

        # ---------------- Compute scores ------------------------
        prev = { } 
        scores = { hk: defaultdict(float) for hk in meta.hotkeys } 
        for crr in RESULTS:
            hk = crr.miner.hotkey
            env = crr.challenge.env.name
            scr = crr.evaluation.score
            if hk in prev:
                prv = prev[ hk ]
                reset = prv.miner.block != crr.miner.block
                reset = prv.miner.model != crr.miner.model
                reset = prv.miner.revision != crr.miner.revision
                if reset:
                    scores[hk][env] = 0
            prev[ hk ] = crr
            if crr.response.success or crr.miner.chute['hot']:
                scores[hk][env] = scr * (1 - ALPHA) + scores[hk][env] * ALPHA

        # ---------------- Compute Pairwise Dominance -----------------
        ranks, counts = {}, defaultdict(int)
        for e in ENVS:
            uniq = sorted({scores[h][e] for h in meta.hotkeys}, reverse=True)
            rank_of = {score: i + 1 for i, score in enumerate(uniq)}
            ranks[e] = {h: rank_of[scores[h][e]] for h in meta.hotkeys}
        for a in meta.hotkeys:
            for b in meta.hotkeys:
                if a == b: continue
                better    = sum(ranks[e][a] < ranks[e][b] for e in ENVS)
                not_worse = all(ranks[e][a] <= ranks[e][b] for e in ENVS)
                if not_worse and better >= 1:
                    counts[a] += 1

        # ---------------- Set weights. ------------------------
        best = max( meta.hotkeys, key=lambda hk: (counts.get(hk, 0), -prev[hk].miner.block if hk in prev else float('inf')) )                
        weights = [1.0 if hk == best else 0.0 for hk in meta.hotkeys]      
        await sub.set_weights(
            wallet=wallet,
            netuid=NETUID,
            uids=meta.uids,
            weights=weights,
            wait_for_inclusion=False
        )
        
        # ---------------- Print State ------------------------
        h = ["UID","Model","Rev"] + [f"{e} Score" for e in ENVS] + [f"{e} Rank" for e in ENVS] + ["Count","Weight"]
        r = [
            [str(m.uid), m.model, m.revision or ""] 
            + [f"{scores[hk][e]:.4f}" for e in ENVS] 
            + [str(ranks[e][hk])      for e in ENVS] 
            + [str(counts.get(hk,0)), f"{weights[meta.hotkeys.index(hk)]:.1f}"]
            for hk in meta.hotkeys
            if (last := prev.get(hk)) and last.miner.model and (m := last.miner)
        ]
        logger.info("\nValidator Summary:\n%s", tabulate(r, h, tablefmt="plain"))
        
    # ---------------- Watchdog for liveness. ------------------------
    LAST_PROGRESS = time.monotonic()

    async def _watchdog(threshold: int = 300, check_interval: int = 60):
        """Terminate the process if no progress event in <threshold> seconds."""
        while True:
            await asyncio.sleep(check_interval)
            if time.monotonic() - LAST_PROGRESS > threshold:
                logger.error("Watchdog: no progress for %ds — exiting so container can restart", threshold)
                os._exit(1)  # rude but ensures container restart

    # ---------------- Main loop. ------------------------
    async def loop():
        nonlocal LAST_PROGRESS
        subtensor = None
        envs = { name: cls() for name, cls in ENVS.items() }
        # Kick off background watchdog
        asyncio.create_task(_watchdog())
        while True:
            try:
                if subtensor is None: subtensor = await get_subtensor()
                meta = await subtensor.metagraph( NETUID )
                blk = await subtensor.get_current_block()
                miners_map = await miners(meta=meta)
                miners_map = await validate_miners_unchanged(miners_map, margin_sec=300)
                challenges = [await e.generate() for e in envs.values()]
                results    = await run(challenges, miners_map, timeout=90)
                await sink( wallet = wallet, block = blk, results = results )
                await maybe_set_weights( subtensor, meta, blk )
                # progress marker
                LAST_PROGRESS = time.monotonic()
            except asyncio.CancelledError: break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
            
    asyncio.run( loop() )
    

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
    miner_map = asyncio.run(miners(uids=uid, no_null=True))
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
def push(model_path: str, existing_repo: str, revision: str, coldkey: str, hotkey: str):
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
    chutes_api_key = get_conf("CHUTES_API_KEY")
    chute_user     = get_conf("CHUTE_USER")
    # TODO: validate API creds, exit gracefully if missing

    # -----------------------------------------------------------------------------
    # 2. Prepare HF repo name
    #    - If --existing-repo provided, use it directly and skip local upload
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
    async def commit_to_chain():
        """Submit the model commitment, retrying if the subnet space quota is full."""
        logger.debug("Preparing on-chain commitment")
        sub     = await get_subtensor()
        payload = json.dumps({"model": repo_name, "revision": revision})
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

    asyncio.run(commit_to_chain())

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
        gpu_count=4,
        min_vram_gb_per_gpu=24,
        include=["4090", "l40s", "a6000_ada"],
        exclude=["h200", "b200", "mi300x"],
    ),
    engine_args=(
        "--trust-remote-code "
        "--tool-call-parser deepseekv3 "
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
        logger.trace(stdout.decode())
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

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
            challenge = await SAT().generate()
            await run(challenges=challenge, miners=[miner])
            await sub.wait_for_block()
            miner = (await miners(netuid=NETUID))[my_uid]
            logger.trace("Checked hot status: %s", (miner.chute or {}).get("hot"))

        logger.debug("Model is now hot and ready")

    asyncio.run(warmup_model())
    logger.debug("Mining setup complete. Model is live!")  


async def get_hf_last_modified(repo_id: str, hf_token: Optional[str] = None) -> Optional[dt.datetime]:
 
    hf_token = hf_token or os.getenv("HF_TOKEN") or None

    def _fetch_info():
        api = HfApi(token=hf_token)
        return api.repo_info(repo_id=repo_id, repo_type="model")
    try:
        info = await asyncio.to_thread(_fetch_info)
    except Exception as exc:
        logger.debug("get_hf_last_modified: impossible to get %s – %s", repo_id, exc)
        return None

    ts = getattr(info, "last_modified", None) or getattr(info, "lastModified", None)

    if isinstance(ts, dt.datetime):
        return ts.astimezone(dt.timezone.utc)
    if isinstance(ts, str):
        try:
            return dt.datetime.fromisoformat(ts.rstrip("Z")) .replace(tzinfo=dt.timezone.utc)
        except Exception:
            pass
    return None

async def get_block_timestamp(block: int, netuid: int = NETUID) -> Optional[dt.datetime]:

    ts_ms = await _block_timestamp_ms(block, netuid)
    if ts_ms is None:
        return None
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc)

async def get_chute_last_modified(model: str) -> Optional[dt.datetime]:
    info = await get_chute(model)
    if not info:
        return None
    ts_raw = (
        info.get("updated_at")
    )
    if ts_raw is None:
        return None
    if isinstance(ts_raw, (int, float)):
        ts_sec = ts_raw / 1000 if ts_raw > 1e12 else ts_raw
        return dt.datetime.fromtimestamp(ts_sec, tz=dt.timezone.utc)

    if isinstance(ts_raw, str):
        try:
            return dt.datetime.fromisoformat(ts_raw.rstrip("Z")) .replace(tzinfo=dt.timezone.utc)
        except Exception:
            try:
                return dt.datetime.fromtimestamp(float(ts_raw), tz=dt.timezone.utc)
            except Exception:
                return None
    if isinstance(ts_raw, dt.datetime):
        return ts_raw.astimezone(dt.timezone.utc)

    return None
        

async def _block_timestamp_ms(block: int, netuid: int = NETUID) -> Optional[int]:
    async def _try_archive() -> Optional[int]:
        try:
            st = bt.async_subtensor("archive")
            await st.initialize()
            block_hash = await st.get_block_hash(block)
            ts_obj = await st.substrate.query("Timestamp", "Now", block_hash=block_hash)
            return int(ts_obj.value)
        except Exception as exc:
            logger.debug("_block_timestamp_ms – archive failed: %s", exc)
            return None

    async def _try_default() -> Optional[int]:
        try:
            st = await get_subtensor()
            block_hash = await st.get_block_hash(block)
            ts_obj = await st.substrate.query("Timestamp", "Now", block_hash=block_hash)
            return int(ts_obj.value)
        except Exception as exc:
            logger.debug("_block_timestamp_ms – default failed: %s", exc)
            return None

    ts_ms = await _try_archive()
    if ts_ms is None:
        ts_ms = await _try_default()
    return ts_ms
        

async def validate_miners_unchanged(miners_dict: Dict[int, Miner], margin_sec: int = 300) -> Dict[int, Miner]:
    if not miners_dict:
        return miners_dict

    async def _is_valid(miner: Miner) -> bool:
        commit_dt = await get_block_timestamp(miner.block) if miner.block else None
        if commit_dt is None:
            return True
        hf_ts, chute_ts = await asyncio.gather(
            get_hf_last_modified(miner.model) if miner.model else None,
            get_chute_last_modified(miner.model) if miner.model else None,
        )
        mods = [ts for ts in (hf_ts, chute_ts) if ts]
        if not mods:
            return True
        latest_mod = max(mods)
        return (latest_mod - commit_dt).total_seconds() <= margin_sec

    flags = await asyncio.gather(*[_is_valid(m) for m in miners_dict.values()])

    invalidated = [(uid, miner) for (uid, miner), ok in zip(miners_dict.items(), flags) if not ok]
    
    return {uid: miner for (uid, miner), ok in zip(miners_dict.items(), flags) if ok}


        

