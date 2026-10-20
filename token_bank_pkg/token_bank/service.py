"""FastAPI service wrapping TokenBankV2 instances (in-memory store)."""

import threading
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import TokenBankV2

app = FastAPI(title="Token Bank V2", version="2.0.0")

_BANKS: Dict[str, TokenBankV2] = {}
_LOCK = threading.Lock()


def _get_bank(bank_id: str) -> TokenBankV2:
    bank = _BANKS.get(bank_id)
    if bank is None:
        raise HTTPException(status_code=404, detail=f"Bank not found: {bank_id}")
    return bank


class CreateBankRequest(BaseModel):
    total_budget: int
    policy: str = "hard"
    gamma: float = 0.8
    theta: float = 0.8
    min_quota: int = 5000
    enable_predictive: bool = True
    enable_borrowing: bool = True
    enable_opl: bool = True
    enable_kv: bool = True
    credit_rate: float = 1.5
    rogue_cutoff: float = 0.10
    enable_pacing: bool = False
    pacing_upfront: float = 0.5
    pacing_k: int = 3
    pacing_max_rounds: int = 5
    adaptive_runaway: bool = False
    runaway_token_fraction: float = 0.35
    runaway_rounds: int = 1


class RegisterAgentsRequest(BaseModel):
    role_map: Dict[str, str]


class CanDispatchRequest(BaseModel):
    agent_id: str
    estimated_tokens: int = 0


class ConsumeRequest(BaseModel):
    agent_id: str
    tokens_in: int
    tokens_out: int
    model: Optional[str] = None
    action: str = "inference"
    kv_cache: int = 0


class OplRequest(BaseModel):
    task_result: dict = Field(default_factory=dict)


class CreditRequest(BaseModel):
    task_result: dict = Field(default_factory=dict)


class ProgressRequest(BaseModel):
    answer: str = ""
    approved: bool = False


@app.post("/banks")
def create_bank(req: CreateBankRequest):
    bank = TokenBankV2(
        total_budget=req.total_budget,
        policy=req.policy,
        gamma=req.gamma,
        rebalance_threshold=req.theta,
        min_quota=req.min_quota,
        enable_predictive=req.enable_predictive,
        enable_borrowing=req.enable_borrowing,
        enable_opl=req.enable_opl,
        enable_kv=req.enable_kv,
        credit_rate=req.credit_rate,
        rogue_cutoff=req.rogue_cutoff,
        enable_pacing=req.enable_pacing,
        pacing_upfront=req.pacing_upfront,
        pacing_k=req.pacing_k,
        pacing_max_rounds=req.pacing_max_rounds,
        adaptive_runaway=req.adaptive_runaway,
        runaway_token_fraction=req.runaway_token_fraction,
        runaway_rounds=req.runaway_rounds,
    )
    bank_id = uuid.uuid4().hex
    with _LOCK:
        _BANKS[bank_id] = bank
    return {"bank_id": bank_id}


@app.post("/banks/{bank_id}/agents")
def register_agents(bank_id: str, req: RegisterAgentsRequest):
    bank = _get_bank(bank_id)
    bank.register_agents(req.role_map)
    bank.allocate()
    return {
        "registered": list(req.role_map.keys()),
        "quotas": {aid: acc.quota for aid, acc in bank.ledger.items()},
    }


@app.post("/banks/{bank_id}/can_dispatch")
def can_dispatch(bank_id: str, req: CanDispatchRequest):
    bank = _get_bank(bank_id)
    allowed, reason = bank.can_dispatch(req.agent_id, req.estimated_tokens)
    return {"allowed": allowed, "reason": reason}


@app.post("/banks/{bank_id}/consume")
def consume(bank_id: str, req: ConsumeRequest):
    bank = _get_bank(bank_id)
    return bank.consume(
        req.agent_id, req.tokens_in, req.tokens_out,
        model=req.model, action=req.action, kv_cache=req.kv_cache,
    )


@app.post("/banks/{bank_id}/rebalance")
def rebalance(bank_id: str):
    bank = _get_bank(bank_id)
    return bank.predictive_rebalance()


@app.post("/banks/{bank_id}/opl")
def update_opl(bank_id: str, req: OplRequest):
    bank = _get_bank(bank_id)
    bank.update_opl(req.task_result)
    return {"updated": bank.opl is not None, "task_count": bank.opl.task_count if bank.opl else 0}


@app.post("/banks/{bank_id}/credit")
def update_credit(bank_id: str, req: CreditRequest):
    bank = _get_bank(bank_id)
    bank.update_credit(req.task_result)
    return {"updated": bank.credit_scorer is not None,
            "scores": bank.credit_scorer.summary() if bank.credit_scorer else {}}


@app.post("/banks/{bank_id}/progress")
def update_progress(bank_id: str, req: ProgressRequest):
    bank = _get_bank(bank_id)
    bank.progress_update(req.answer, req.approved)
    return {"runaway_active": bank.runaway_active,
            "released": bank.released,
            "counter": bank._runaway_counter}


@app.get("/banks/{bank_id}/summary")
def summary(bank_id: str):
    bank = _get_bank(bank_id)
    return bank.get_summary()
