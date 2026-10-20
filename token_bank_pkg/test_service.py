"""Smoke test for the Token Bank V2 FastAPI service.

Runs a full flow through fastapi.testclient.TestClient:
create bank -> register 3 agents + allocate -> can_dispatch -> consume past
quota (triggering inter-agent borrow) -> rebalance -> OPL update -> summary.

Runnable directly (python test_service.py) or via pytest.
"""

from fastapi.testclient import TestClient

from token_bank.service import app

ROLE_MAP = {
    "planner": "Planner",
    "executor": "Executor",
    "reviewer": "Reviewer",
}


def test_full_flow():
    client = TestClient(app)

    # 1. Create a bank
    r = client.post("/banks", json={
        "total_budget": 90000, "policy": "hard", "gamma": 0.8,
        "theta": 0.8, "min_quota": 5000,
        "enable_predictive": True, "enable_borrowing": True,
        "enable_opl": True, "enable_kv": True,
    })
    assert r.status_code == 200, r.text
    bank_id = r.json()["bank_id"]
    assert bank_id

    # 2. Register 3 agents (also allocates quotas)
    r = client.post(f"/banks/{bank_id}/agents", json={"role_map": ROLE_MAP})
    assert r.status_code == 200, r.text
    quotas = r.json()["quotas"]
    assert set(quotas) == set(ROLE_MAP)
    assert all(q > 0 for q in quotas.values())

    # 3. can_dispatch within quota
    r = client.post(f"/banks/{bank_id}/can_dispatch",
                    json={"agent_id": "executor", "estimated_tokens": 1000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["allowed"] is True and body["reason"] == "OK"

    # 4. Consume most of executor's quota
    r = client.post(f"/banks/{bank_id}/consume",
                    json={"agent_id": "executor", "tokens_in": 20000,
                          "tokens_out": 4000, "kv_cache": 0})
    assert r.status_code == 200, r.text
    assert r.json()["allowed"] is True

    # 5. can_dispatch past remaining quota -> triggers inter-agent borrow
    r = client.post(f"/banks/{bank_id}/can_dispatch",
                    json={"agent_id": "executor", "estimated_tokens": 15000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["allowed"] is True
    assert "borrowing" in body["reason"], body

    # 6. Consume within the borrowed credit line
    r = client.post(f"/banks/{bank_id}/consume",
                    json={"agent_id": "executor", "tokens_in": 8000,
                          "tokens_out": 2000})
    assert r.status_code == 200, r.text
    assert r.json()["allowed"] is True

    # 7. Consume far beyond quota + credit line -> hard policy rejects
    r = client.post(f"/banks/{bank_id}/consume",
                    json={"agent_id": "executor", "tokens_in": 30000,
                          "tokens_out": 5000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["allowed"] is False
    assert "Quota exceeded" in body["reason"]

    # 8. Predictive rebalance
    r = client.post(f"/banks/{bank_id}/rebalance")
    assert r.status_code == 200, r.text
    assert "performed" in r.json()

    # 9. OPL update with a task result
    r = client.post(f"/banks/{bank_id}/opl", json={"task_result": {
        "agent_calls": [
            {"agent_id": "planner", "success": True},
            {"agent_id": "executor", "success": False, "block_reason": "budget exceeded"},
        ]
    }})
    assert r.status_code == 200, r.text
    assert r.json()["updated"] is True

    # 10. Summary: quota/used/borrowed present and sane
    r = client.get(f"/banks/{bank_id}/summary")
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["total_budget"] == 90000
    assert s["total_used"] > 0
    assert s["total_borrowed"] > 0
    assert s["borrow_count"] >= 1
    assert 0 <= s["reserve"] < 90000
    assert set(s["agents"]) == set(ROLE_MAP)
    for aid, a in s["agents"].items():
        assert {"quota", "used", "borrowed"} <= set(a), (aid, a)
        assert a["quota"] > 0
        assert a["used"] >= 0
        assert a["borrowed"] >= 0
    ex = s["agents"]["executor"]
    assert ex["borrowed"] > 0
    assert ex["used"] > ex["quota"] - ex["borrowed"]  # ate into the credit line

    # 11. Unknown bank_id -> 404 on every endpoint
    assert client.get("/banks/nope/summary").status_code == 404
    assert client.post("/banks/nope/rebalance").status_code == 404
    assert client.post("/banks/nope/agents", json={"role_map": ROLE_MAP}).status_code == 404
    assert client.post("/banks/nope/can_dispatch",
                       json={"agent_id": "x", "estimated_tokens": 1}).status_code == 404
    assert client.post("/banks/nope/consume",
                       json={"agent_id": "x", "tokens_in": 1, "tokens_out": 1}).status_code == 404
    assert client.post("/banks/nope/opl", json={"task_result": {}}).status_code == 404


if __name__ == "__main__":
    test_full_flow()
    print("SMOKE TEST PASSED: full bank flow (create/register/consume/borrow/rebalance/opl/summary/404)")
