"""Core engine extracted from _run_v2_experiment.py (NOTEBOOK CELL 6 + CELL 7).

Behavior is preserved; only decoupling changes were made:
- KVCacheTracker.summarize_context: tiktoken is imported lazily; when it is
  unavailable, token counts fall back to a len(text) // 4 estimate.
- TokenBankV2: PRICING / PRIMARY_MODEL / THROTTLE_MODEL globals became
  __init__ parameters (pricing, primary_model, throttle_model) with generic
  gpt-4o / gpt-4o-mini defaults.
- PredictiveRebalancer keeps the statsmodels ARIMA try/except -> EWMA fallback.
- Budget pacing and stall-detection hooks from the latest experiment script are
  included and can be enabled via enable_pacing.
"""

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

# Try to import statsmodels for ARIMA; fallback to simple EWMA if unavailable
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Default pricing (USD per 1M tokens) and model tiers. These replace the
# experiment script's deepseek-specific globals with generic defaults.
DEFAULT_PRICING = {
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
}
DEFAULT_PRIMARY_MODEL = 'gpt-4o'
DEFAULT_THROTTLE_MODEL = 'gpt-4o-mini'

# Lazily-initialized tiktoken encoder (None when tiktoken is unavailable).
_ENCODER = None
_ENCODER_RESOLVED = False


def _count_tokens(text: str) -> int:
    """Precise token count via tiktoken when installed; len(text)//4 otherwise."""
    global _ENCODER, _ENCODER_RESOLVED
    if not _ENCODER_RESOLVED:
        _ENCODER_RESOLVED = True
        try:
            import tiktoken
            _ENCODER = tiktoken.encoding_for_model('gpt-4o')
        except Exception:
            _ENCODER = None
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text) // 4


# ============ (from NOTEBOOK CELL 6) ============
@dataclass
class AgentAccountV2:
    agent_id: str
    role: str
    quota: int = 0
    used: int = 0
    priority: int = 2
    borrowed: int = 0       # credit line utilized
    lent: int = 0           # credit extended to others
    history: List[Dict] = field(default_factory=list)
    consumption_log: deque = field(default_factory=lambda: deque(maxlen=100))
    kv_cache_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.quota + self.borrowed - self.used)

    @property
    def utilization(self) -> float:
        denom = self.quota + self.borrowed
        return self.used / denom if denom > 0 else 0.0

    @property
    def effective_tokens(self) -> int:
        """KV-cache aware: prompt + 0.3 * kv_cache"""
        return self.used + int(0.3 * self.kv_cache_tokens)

    @property
    def total_cost(self) -> float:
        return sum(h.get('cost_usd', 0) for h in self.history)


class BudgetAdapter:
    """Wraps pre-execution optimizer outputs into initial quotas."""
    def __init__(self, external_plan: Optional[Dict] = None):
        self.plan = external_plan or {}

    def get_initial_quotas(self, total_budget: int, n_agents: int,
                           role_map: Dict[str, str]) -> Dict[str, int]:
        if not self.plan:
            return {aid: total_budget // n_agents for aid in role_map}
        quotas = {}
        for aid, role in role_map.items():
            q = self.plan.get(aid, self.plan.get(role, total_budget // n_agents))
            quotas[aid] = int(q)
        return quotas

    def compute_deviation(self, planned: int, actual: int) -> float:
        return abs(actual - planned) / max(planned, 1)


class PredictiveRebalancer:
    """ARIMA/EWMA forecast + opportunity cost-driven rebalancing."""
    def __init__(self, horizon: int = 2, min_history: int = 1):
        self.horizon = horizon
        self.min_history = min_history

    def forecast(self, history: deque) -> Tuple[float, float]:
        """Returns (expected_consumption_over_horizon, trend). Agents with no
        history forecast 0, i.e. their unused quota counts as surplus."""
        if len(history) < self.min_history:
            return 0.0, 0.0
        series = list(history)
        if HAS_STATSMODELS and len(series) >= 5:
            try:
                model = ARIMA(series, order=(1,1,1))
                fitted = model.fit()
                fcast = fitted.forecast(steps=self.horizon)
                expected = sum(max(0, x) for x in fcast)
                trend = float(fcast[-1] - fcast[0]) if self.horizon > 1 else 0.0
                return expected, trend
            except:
                pass
        # EWMA fallback
        alpha = 0.3
        ewma = series[0]
        for x in series[1:]:
            ewma = alpha * x + (1 - alpha) * ewma
        expected = ewma * self.horizon
        trend = series[-1] - series[0]
        return expected, trend

    def should_rebalance(self, agents: List[AgentAccountV2],
                         reserve: int, threshold: float = 0.8) -> Tuple[bool, Dict]:
        """Opportunity cost-based rebalancing decision.

        threshold (theta) is the required headroom: an agent is predicted
        deficient when its expected near-term consumption exceeds `threshold`
        of its remaining quota. Lower theta -> triggers earlier."""
        forecasts = {}
        for a in agents:
            exp, trend = self.forecast(a.consumption_log)
            forecasts[a.agent_id] = {'expected': exp, 'trend': trend,
                                      'predicted_surplus': a.remaining - exp}

        deficit_agents = [a for a in agents
                          if forecasts[a.agent_id]['expected'] > threshold * a.remaining]
        surplus_agents = [a for a in agents if forecasts[a.agent_id]['predicted_surplus'] > 0]

        if not deficit_agents or not surplus_agents:
            return False, {}

        cost_miss = sum(a.priority * forecasts[a.agent_id]['expected']
                        for a in deficit_agents)
        reclaimable = sum((a.quota - a.used) * 0.3 for a in surplus_agents if a.utilization < 0.5)
        cost_rebal = reclaimable * 0.05

        return cost_miss > cost_rebal, forecasts


class CreditScorer:
    """Bank-style credit scoring: tracks whether borrowed credit converts to
    task success ('repayment') per role, and sizes/prices credit accordingly.
    Thread-safe so it can be shared across concurrent task banks."""
    def __init__(self, base_rate: float = 0.5, base_interest: float = 0.15):
        self.base_rate = base_rate
        self.base_interest = base_interest
        self.history: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()

    def score(self, role: str) -> float:
        with self._lock:
            h = self.history.get(role, {'alpha': 1.0, 'beta': 1.0})
            return h['alpha'] / (h['alpha'] + h['beta'])

    def credit_rate_for(self, role: str) -> float:
        """score 0 -> 0.5x base_rate, score 1 -> 1.5x base_rate."""
        return self.base_rate * (0.5 + self.score(role))

    def interest_for(self, role: str) -> float:
        """Risk pricing: score 0 -> 2x base interest, score 1 -> 0.5x."""
        return self.base_interest * (1.5 - self.score(role))

    def record(self, role: str, repaid: bool) -> None:
        with self._lock:
            h = self.history.setdefault(role, {'alpha': 1.0, 'beta': 1.0})
            h['alpha' if repaid else 'beta'] += 1

    def summary(self) -> Dict[str, float]:
        with self._lock:
            return {r: round(self.score(r), 3) for r in self.history}


class BudgetBorrowing:
    """Inter-agent credit market with safety guarantees (Theorem 2)."""
    def __init__(self, credit_rate: float = 0.10, interest_rate: float = 0.15,
                 scorer: Optional['CreditScorer'] = None, rogue_cutoff: float = 0.10):
        self.credit_rate = credit_rate
        self.interest_rate = interest_rate
        self.scorer = scorer
        # Option C: credit score sizes/prices credit lines and hard-cuts confirmed
        # rogues, but does not gate normal peer lending decisions.
        self.rogue_cutoff = rogue_cutoff
        self.loans: List[Dict] = []

    def max_credit_line(self, agent: AgentAccountV2) -> int:
        rate = self.scorer.credit_rate_for(agent.role) if self.scorer else self.credit_rate
        return int(agent.quota * rate)

    def can_borrow(self, borrower: AgentAccountV2, amount: int,
                   lender: AgentAccountV2, reserve: int) -> bool:
        """Theorem 2 safety check + rogue-only cutoff."""
        if amount <= 0:
            return False
        if self.scorer:
            score = self.scorer.score(borrower.role)
            if score < self.rogue_cutoff:
                return False
        if borrower.borrowed + amount > self.max_credit_line(borrower):
            return False
        if lender.priority > borrower.priority:
            return False
        if lender.remaining < amount:
            return False
        if reserve <= 0:
            return False
        return True

    def execute_loan(self, borrower: AgentAccountV2, lender: AgentAccountV2,
                     amount: int) -> Dict:
        rate = self.scorer.interest_for(borrower.role) if self.scorer else self.interest_rate
        interest = int(amount * rate)
        borrower.borrowed += amount
        lender.lent += amount + interest
        lender.quota -= amount
        self.loans.append({
            'borrower': borrower.agent_id, 'lender': lender.agent_id,
            'amount': amount, 'interest': interest, 'timestamp': time.time()
        })
        return {'success': True, 'amount': amount, 'interest': interest}


class OnlinePriorityLearner:
    """Thompson Sampling for adaptive role priorities."""
    def __init__(self, roles: List[str], cold_start_tasks: int = 10):
        self.roles = roles
        self.posteriors = {r: {'alpha': 1.0, 'beta': 1.0} for r in roles}
        self.cold_start = cold_start_tasks
        self.task_count = 0
        self.manual_priorities = {'Planner': 3, 'Executor': 2, 'Reviewer': 2, 'Debugger': 1}

    def get_priorities(self) -> Dict[str, int]:
        if self.task_count < self.cold_start:
            return self.manual_priorities.copy()
        priorities = {}
        for r in self.roles:
            p = beta_dist.rvs(self.posteriors[r]['alpha'], self.posteriors[r]['beta'])
            priorities[r] = max(1, min(3, round(1 + 2 * p)))
        return priorities

    def update(self, role_success: Dict[str, bool], role_blocked: Dict[str, bool]) -> None:
        self.task_count += 1
        for role in self.roles:
            if role_success.get(role, False):
                self.posteriors[role]['alpha'] += 1
            elif role_blocked.get(role, False):
                self.posteriors[role]['beta'] += 1

    def get_regret_bound(self, T: int) -> float:
        K = len(self.roles)
        return math.sqrt(K * T * math.log(T + 1))


class KVCacheTracker:
    """Tracks KV-cache accumulation and triggers summarization."""
    def __init__(self, compression_factor: float = 0.3, trigger_ratio: float = 0.85):
        self.compression_factor = compression_factor
        self.trigger_ratio = trigger_ratio
        self.summarization_count = 0

    def effective_tokens(self, prompt_tokens: int, kv_cache: int) -> int:
        return prompt_tokens + int(self.compression_factor * kv_cache)

    def should_summarize(self, agent: AgentAccountV2) -> bool:
        eff = agent.effective_tokens
        return eff / max(agent.quota, 1) > self.trigger_ratio

    def summarize_context(self, messages: List[Dict], llm_client) -> Tuple[List[Dict], int, int]:
        self.summarization_count += 1
        summary_msg = {
            'role': 'system',
            'content': '[Context summarized due to KV-cache pressure. Prior conversation condensed.]'
        }
        kept = messages[-2:] if len(messages) > 2 else messages
        new_messages = [summary_msg] + kept
        tokens_in = sum(_count_tokens(m.get('content', '')) for m in new_messages)
        return new_messages, tokens_in, 0


# ============ (from NOTEBOOK CELL 7) ============
class TokenBankV2:
    """Adaptive Runtime Budget Governance Middleware."""
    def __init__(self, total_budget: int, policy='hard', external_plan=None,
                 gamma=0.8, rebalance_threshold=0.8, enable_predictive=True,
                 enable_borrowing=True, enable_opl=True, enable_kv=True,
                 min_quota=5000, anomaly_sigma=3.0, shared_opl=None,
                 shared_credit_scorer=None, rogue_cutoff=0.10,
                 credit_rate=0.5, enable_pacing=False,
                 pacing_upfront=0.5, pacing_k=3, pacing_max_rounds=5,
                 adaptive_runaway=False, runaway_token_fraction=0.35,
                 runaway_rounds=1, runaway_min_tokens=0,
                 pricing=None, primary_model=DEFAULT_PRIMARY_MODEL,
                 throttle_model=DEFAULT_THROTTLE_MODEL):
        self.total_budget = total_budget
        self.reserve = total_budget
        self.policy = policy
        self.gamma = gamma
        self.theta = rebalance_threshold
        self.q_min = min_quota
        self.anomaly_sigma = anomaly_sigma
        self.adapter = BudgetAdapter(external_plan)
        self.predictor = PredictiveRebalancer(horizon=2) if enable_predictive else None
        self.borrowing = BudgetBorrowing(
            credit_rate=credit_rate, scorer=shared_credit_scorer,
            rogue_cutoff=rogue_cutoff
        ) if enable_borrowing else None
        self.credit_scorer = shared_credit_scorer
        self.rogue_cutoff = rogue_cutoff
        # shared_opl: an OnlinePriorityLearner shared across tasks so priority
        # learning accumulates beyond a single task (else cold-start never ends)
        self.opl = shared_opl
        self.enable_opl = enable_opl
        self.kv_tracker = KVCacheTracker() if enable_kv else None
        # Budget pacing (soft segmentation): the first `pacing_upfront` fraction
        # of the budget is available upfront; the rest is split into `pacing_k`
        # segments that are released only when (a) >=80% of the released pool
        # is consumed and (b) a Reviewer checkpoint has completed since the
        # last release. Global ceiling independent of per-agent quota/borrowing.
        self.pacing = enable_pacing
        self.pacing_k = pacing_k
        self.pacing_max_rounds = pacing_max_rounds
        # Adaptive runaway guard: pacing/stall are dormant until the task is both
        # stuck (answer unchanged) AND has burned a significant fraction of budget.
        self.adaptive_runaway = adaptive_runaway
        self.runaway_token_fraction = runaway_token_fraction
        self.runaway_rounds = runaway_rounds
        # Absolute floor for the runaway trigger: the guard fires only after
        # max(frac * budget, runaway_min_tokens) tokens are burned, so very
        # tight budgets (where frac * budget is tiny) do not over-kill tasks.
        self.runaway_min_tokens = runaway_min_tokens
        self.runaway_active = False
        self._runaway_counter = 0
        self._last_answer = None
        if self.adaptive_runaway:
            self.released = total_budget
        else:
            self.released = int(total_budget * pacing_upfront)
        # ceil so the final release tops up to exactly total_budget
        self.segment_size = -(-(total_budget - self.released) // self.pacing_k)
        self._reviews_since_release = 0
        self.pacing_releases = 0
        self.ledger = {}
        self.rebalance_count = 0
        self.borrow_count = 0
        self.anomaly_log = []
        self.trajectory = []
        self._velocity_window = defaultdict(list)
        # Pricing / model-tier configuration (formerly module globals).
        self.pricing = pricing if pricing is not None else dict(DEFAULT_PRICING)
        self.primary_model = primary_model
        self.throttle_model = throttle_model

    def total_used(self):
        return sum(a.used for a in self.ledger.values())

    def progress_update(self, answer: str, approved: bool, made_progress: bool = False):
        """Runner reports per-round outcome. Adaptive runaway guard activates
        only when the task is stuck (answer unchanged) AND has burned enough
        budget. Approved tasks or tasks that are still changing answers keep
        the full budget. `made_progress=True` (e.g. the round executed a code
        tool) counts as progress even if the textual answer is unchanged."""
        if not self.adaptive_runaway:
            return
        if approved or made_progress:
            self.runaway_active = False if approved else self.runaway_active
            self._runaway_counter = 0
            self._last_answer = None
            return
        norm = ' '.join(answer.lower().split()) if answer else ''
        if self._last_answer is not None and norm == self._last_answer:
            self._runaway_counter += 1
        else:
            self._runaway_counter = 0
            self._last_answer = norm
        if (self._runaway_counter >= self.runaway_rounds and
                self.total_used() >= max(self.runaway_token_fraction * self.total_budget,
                                         self.runaway_min_tokens)):
            if not self.runaway_active:
                self.runaway_active = True
                self.released = min(
                    self.total_budget,
                    self.total_used() + int((self.total_budget - self.total_used()) * 0.3)
                )
                remaining = self.total_budget - self.released
                self.segment_size = -(-remaining // self.pacing_k) if remaining > 0 else 0
                self._log_event('RUNAWAY_GUARD', {
                    'released': self.released,
                    'counter': self._runaway_counter,
                    'token_fraction': self.total_used() / max(self.total_budget, 1),
                })

    def pacing_release_check(self, action):
        """Called after every successful consume(). Releases the next pacing
        segment when the released pool is >=80% consumed AND a Reviewer
        checkpoint happened since the last release."""
        if not self.pacing or self.released >= self.total_budget:
            return
        if action == 'review':
            self._reviews_since_release += 1
        if (self.total_used() >= 0.8 * self.released
                and self._reviews_since_release >= 1):
            self.released = min(self.total_budget, self.released + self.segment_size)
            self.pacing_releases += 1
            self._reviews_since_release = 0
            self._log_event('PACING_RELEASE', {'released': self.released})

    def register_agents(self, role_map, priorities=None):
        for aid, role in role_map.items():
            p = priorities.get(role, 2) if priorities else 2
            self.ledger[aid] = AgentAccountV2(agent_id=aid, role=role, priority=p)
        if self.enable_opl and self.opl is None:
            self.opl = OnlinePriorityLearner(list(set(role_map.values())))

    def allocate(self, priority_weight=True):
        n = len(self.ledger)
        if n == 0:
            return
        role_map = {aid: acc.role for aid, acc in self.ledger.items()}
        plan_quotas = self.adapter.get_initial_quotas(self.total_budget, n, role_map)
        if self.opl and self.opl.task_count >= self.opl.cold_start:
            learned_p = self.opl.get_priorities()
        else:
            learned_p = {acc.role: acc.priority for acc in self.ledger.values()}
        total_p = sum(learned_p.get(acc.role, 2) for acc in self.ledger.values())
        available = self.total_budget - n * self.q_min
        runtime_quotas = {}
        for aid, acc in self.ledger.items():
            p = learned_p.get(acc.role, 2)
            runtime_quotas[aid] = max(self.q_min, int(p * available / total_p))
        for aid in self.ledger:
            q_plan = plan_quotas.get(aid, self.total_budget // n)
            q_runtime = runtime_quotas.get(aid, self.total_budget // n)
            self.ledger[aid].quota = int(self.gamma * q_plan + (1 - self.gamma) * q_runtime)
        self._log_event('ALLOCATE', {k: self.ledger[k].quota for k in self.ledger})

    def can_dispatch(self, agent_id, estimated_tokens=0):
        if agent_id not in self.ledger:
            return False, 'Agent not registered'
        # Pacing gate: global ceiling on task-level consumption, independent of
        # per-agent quota and borrowing. For adaptive runaway mode the gate is
        # dormant until the task is flagged as runaway; normal converging tasks
        # keep the full budget.
        pacing_active = self.pacing and (not self.adaptive_runaway or self.runaway_active)
        if pacing_active and self.total_used() + estimated_tokens > self.released:
            return False, f'Pacing gate: {self.total_used()}/{self.released} awaiting checkpoint'
        acc = self.ledger[agent_id]
        grace = 1.0
        if self.policy == 'grace':
            grace = {3: 1.10, 2: 1.05, 1: 1.00}.get(acc.priority, 1.0)
        effective_quota = int((acc.quota + acc.borrowed) * grace)
        if acc.used + estimated_tokens > effective_quota:
            if self.borrowing and self._try_borrow(agent_id, estimated_tokens - acc.remaining):
                return True, 'OK (after borrowing)'
            return False, f'Quota exceeded: {acc.used}/{effective_quota}'
        if estimated_tokens > self.reserve:
            return False, f'Global reserve depleted: {self.reserve}'
        return True, 'OK'

    def _try_borrow(self, borrower_id, amount):
        if not self.borrowing or amount <= 0:
            return False
        borrower = self.ledger[borrower_id]
        candidates = [a for a in self.ledger.values()
                      if a.agent_id != borrower_id and a.remaining > amount]
        # Option C: normal lending decisions are not gated by credit score.
        candidates.sort(key=lambda a: (a.priority, -a.remaining))
        for lender in candidates:
            if self.borrowing.can_borrow(borrower, amount, lender, self.reserve):
                result = self.borrowing.execute_loan(borrower, lender, amount)
                self.borrow_count += 1
                self._log_event('BORROW', result)
                return True
        return False

    def consume(self, agent_id, tokens_in, tokens_out, model=None,
                action='inference', kv_cache=0):
        if agent_id not in self.ledger:
            return {'allowed': False, 'reason': 'Agent not registered'}
        if model is None:
            model = self.primary_model
        acc = self.ledger[agent_id]
        acc.kv_cache_tokens += kv_cache
        if self.kv_tracker:
            total_tokens = self.kv_tracker.effective_tokens(tokens_in, acc.kv_cache_tokens) + tokens_out
        else:
            total_tokens = tokens_in + tokens_out
        cost = self._compute_cost(tokens_in, tokens_out, model)
        if self.policy == 'throttle' and acc.utilization > 0.9:
            model = self.throttle_model
            cost = self._compute_cost(tokens_in, tokens_out, model)
            self._log_event('THROTTLE', {'agent': agent_id, 'new_model': model})
        grace = 1.0
        if self.policy == 'grace':
            grace = {3: 1.10, 2: 1.05, 1: 1.00}.get(acc.priority, 1.0)
        effective_quota = int((acc.quota + acc.borrowed) * grace)
        if acc.used + total_tokens > effective_quota:
            allowed_tokens = max(0, effective_quota - acc.used)
            if self.policy == 'hard' or allowed_tokens == 0:
                return {'allowed': False, 'actual_in': 0, 'actual_out': 0,
                        'reason': f'Quota exceeded: {acc.used}/{effective_quota}'}
            total_tokens = allowed_tokens
            tokens_out = min(tokens_out, allowed_tokens)
            tokens_in = allowed_tokens - tokens_out
        if total_tokens > self.reserve:
            return {'allowed': False, 'actual_in': 0, 'actual_out': 0,
                    'reason': f'Global reserve depleted: {self.reserve}'}
        acc.used += total_tokens
        self.reserve -= total_tokens
        acc.consumption_log.append(total_tokens)
        record = {'timestamp': datetime.now().isoformat(), 'tokens_in': tokens_in,
                  'tokens_out': tokens_out, 'kv_cache': kv_cache, 'model': model,
                  'cost_usd': cost, 'action': action}
        acc.history.append(record)
        self._log_event('CONSUME', {'agent': agent_id, 'tokens': total_tokens, 'cost': cost})
        self._check_anomaly(agent_id, total_tokens)
        self.pacing_release_check(action)
        return {'allowed': True, 'actual_in': tokens_in, 'actual_out': tokens_out,
                'model': model, 'cost_usd': cost, 'reason': 'OK'}

    def predictive_rebalance(self):
        if not self.predictor:
            return {'performed': False, 'reason': 'Disabled'}
        agents = list(self.ledger.values())
        should, forecasts = self.predictor.should_rebalance(agents, self.reserve, self.theta)
        if not should:
            return {'performed': False, 'reason': 'Opportunity cost too low', 'forecasts': forecasts}
        surplus = [a for a in agents if forecasts[a.agent_id]['predicted_surplus'] > 0 and a.utilization < 0.5]
        deficit = [a for a in agents if forecasts[a.agent_id]['expected'] > self.theta * a.remaining]
        if not surplus or not deficit:
            return {'performed': False, 'reason': 'No matching pairs', 'forecasts': forecasts}
        reclaimable = int(sum((a.quota - a.used) * 0.3 for a in surplus))
        if reclaimable <= 0:
            return {'performed': False, 'reason': 'No reclaimable quota'}
        # Option C: rebalancing uses role priority only, not credit score.
        deficit.sort(key=lambda a: a.priority, reverse=True)
        total_p = sum(a.priority for a in deficit) or 1
        transfers = []
        for a in deficit:
            share = int(reclaimable * (a.priority / total_p))
            if share > 0:
                a.quota += share
                transfers.append({'to': a.agent_id, 'amount': share})
        total_surplus_rem = sum(a.remaining for a in surplus) or 1
        for a in surplus:
            deduction = int(reclaimable * (a.remaining / total_surplus_rem))
            a.quota = max(self.q_min, a.quota - deduction)
        self.rebalance_count += 1
        self._log_event('PREDICTIVE_REBALANCE', {'transfers': transfers, 'total': reclaimable})
        return {'performed': True, 'transfers': transfers, 'total': reclaimable, 'forecasts': forecasts}

    def update_opl(self, task_result):
        if not self.opl:
            return
        role_success, role_blocked = {}, {}
        for call in task_result.get('agent_calls', []):
            aid = call.get('agent_id', '')
            if aid in self.ledger:
                role = self.ledger[aid].role
                role_success[role] = call.get('success', False)
                role_blocked[role] = 'budget' in str(call.get('block_reason', '')).lower()
        self.opl.update(role_success, role_blocked)

    def update_credit(self, task_result):
        """Record credit repayment: an agent that borrowed and saw the task
        succeed gets a positive record; borrowed-but-failed is a default."""
        if not self.credit_scorer:
            return
        for acc in self.ledger.values():
            if acc.borrowed > 0:
                self.credit_scorer.record(acc.role, bool(task_result.get('success')))

    def _check_anomaly(self, agent_id, tokens):
        now = time.time()
        window = self._velocity_window[agent_id]
        window.append((now, tokens))
        window[:] = [(t, tok) for t, tok in window if now - t < 60]
        if len(window) < 3:
            return
        velocities = [tok for _, tok in window]
        mean_v, std_v = np.mean(velocities), np.std(velocities)
        if std_v > 0 and tokens > mean_v + self.anomaly_sigma * std_v:
            self.anomaly_log.append({'agent': agent_id, 'timestamp': now,
                                      'tokens': tokens, 'mean_v': mean_v, 'std_v': std_v})
            self._log_event('ANOMALY', {'agent': agent_id, 'tokens': tokens})

    def _compute_cost(self, tokens_in, tokens_out, model):
        p = self.pricing.get(model, self.pricing[self.primary_model])
        return (tokens_in * p['input'] + tokens_out * p['output']) / 1e6

    def _log_event(self, event_type, data):
        self.trajectory.append({'type': event_type, 'time': time.time(), 'data': data, 'reserve': self.reserve})

    def get_summary(self):
        total_used = sum(a.used for a in self.ledger.values())
        total_cost = sum(a.total_cost for a in self.ledger.values())
        total_borrowed = sum(a.borrowed for a in self.ledger.values())
        return {
            'total_budget': self.total_budget, 'reserve': self.reserve,
            'total_used': total_used, 'total_cost_usd': round(total_cost, 4),
            'total_borrowed': total_borrowed, 'rebalance_count': self.rebalance_count,
            'borrow_count': self.borrow_count, 'anomaly_count': len(self.anomaly_log),
            'summarization_count': self.kv_tracker.summarization_count if self.kv_tracker else 0,
            'pacing_releases': self.pacing_releases,
            'runaway_active': self.runaway_active,
            'agents': {k: {'quota': v.quota, 'used': v.used, 'borrowed': v.borrowed,
                          'utilization': round(v.utilization, 3), 'cost_usd': round(v.total_cost, 4)}
                     for k, v in self.ledger.items()}
        }
