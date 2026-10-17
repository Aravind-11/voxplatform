"""Model and voice A/B: stable assignment, agent-save validation, per-call model wrapper."""
import numpy as np

from api.experiments import choose_variant
from api.runtime import Runtime, _count
from api.store import Store
from core.metrics import METRICS
from tests.unit.test_api import FakeTTS, FakeVAD, make, tenant

EXP = {"id": "exp1", "variants": [
    {"id": "A", "weight": 50, "voice": "af_heart", "systemPrompt": "Prompt A.", "model": "ultravox"},
    {"id": "B", "weight": 50, "voice": "af_bella", "model": "ultravox-b"}]}
SPLIT = {"id": "exp1", "variants": [{"id": "A", "weight": 1}, {"id": "B", "weight": 99}]}


class T:
    out_rate = 8000

    async def send_audio(self, pcm, rate=8000):
        return None

    async def clear(self):
        return None

    async def end(self):
        return None


class SharedLLM:
    def __init__(self):
        self.model = "base-model"
        self.marker = "shared"
        self.seen = []

    async def stream(self, history, audio, followups, tools):
        self.seen.append((self.model, self.marker))
        yield "ok"

    async def transcribe(self, audio):
        return "hi"


def test_choose_variant_is_deterministic_and_respects_a_1_99_split():
    n = 2000
    counts = {"A": 0, "B": 0}
    sample = {}
    for i in range(n):
        cid = f"call_{i}"
        vid = choose_variant(SPLIT, cid)["id"]
        counts[vid] += 1
        sample.setdefault(vid, cid)
    assert set(sample) == {"A", "B"}
    assert counts["A"] < n * 0.1 and counts["B"] > n * 0.9
    for vid, cid in sample.items():
        assert choose_variant(SPLIT, cid)["id"] == vid
        assert choose_variant(SPLIT, cid) == choose_variant(SPLIT, cid)


def test_agent_save_rejects_a_zero_weight_and_keeps_a_valid_experiment():
    c, store, _ = make()
    key, _ = tenant(c)
    h = {"X-API-Key": key}
    zero = {"id": "exp1", "variants": [{"id": "A", "weight": 0, "voice": "af_heart"},
                                       {"id": "B", "weight": 99, "voice": "af_bella"}]}
    assert c.post("/v1/agents", json={"name": "n", "experiment": zero}, headers=h).status_code == 422
    dup = {"id": "exp1", "variants": [{"id": "A", "weight": 1}, {"id": "A", "weight": 1}]}
    assert c.post("/v1/agents", json={"name": "n", "experiment": dup}, headers=h).status_code == 422
    too_many = {"id": "exp1", "variants": [{"id": str(i), "weight": 1} for i in range(5)]}
    assert c.post("/v1/agents", json={"name": "n", "experiment": too_many}, headers=h).status_code == 422
    bad_voice = {"id": "exp1", "variants": [{"id": "A", "weight": 1, "voice": "not_a_voice"}]}
    assert c.post("/v1/agents", json={"name": "n", "experiment": bad_voice}, headers=h).status_code == 422
    saved = c.post("/v1/agents", json={"name": "n", "systemPrompt": "base", "experiment": EXP}, headers=h)
    assert saved.status_code == 200
    assert saved.json()["config"]["experiment"]["id"] == "exp1"
    assert c.patch(f"/v1/agents/{saved.json()['id']}", json={"experiment": zero}, headers=h).status_code == 422
    assert store.get_agent_any(saved.json()["id"])["config"]["experiment"]["variants"][0]["weight"] == 50


async def test_build_session_records_the_variant_and_does_not_mutate_the_shared_model():
    c, store, _ = make()
    key, tid = tenant(c)
    h = {"X-API-Key": key}
    agent = c.post("/v1/agents", json={"name": "n", "systemPrompt": "base prompt", "voice": "af_heart",
                                       "experiment": EXP}, headers=h)
    assert agent.status_code == 200
    created = c.post("/v1/calls", json={"agentId": agent.json()["id"]}, headers=h).json()
    call = store.get_call_any(created["id"])
    llm = SharedLLM()
    voices = []
    rt = Runtime(llm, lambda voice: voices.append(voice) or FakeTTS(), FakeVAD, store)
    sess = rt.build_session(call, T())
    chosen = choose_variant(call["config"]["experiment"], call["id"])
    stored = store.get_call(tid, call["id"])
    assert stored["metadata"]["experiment"] == "exp1"
    assert stored["metadata"]["variant"] == chosen["id"]
    assert voices == [chosen["voice"]]
    if chosen.get("systemPrompt"):
        assert sess.history[0]["content"] == chosen["systemPrompt"]
    else:
        assert sess.history[0]["content"] == "base prompt"
    async for _ in sess.llm.stream([], np.zeros(0, np.int16), [], []):
        pass
    assert llm.model == "base-model"
    assert llm.seen == [(chosen["model"], "shared")]
    plain = store.create_call(tid, None, "webSocket", {"systemPrompt": "plain"}, {})
    sess2 = rt.build_session(plain, T())
    assert sess2.llm is llm
    assert "experiment" not in store.get_call_any(plain["id"])["metadata"]
    report = c.get("/v1/experiments/exp1", headers=h)
    assert report.status_code == 200
    counts = {row["id"]: row["calls"] for row in report.json()["variants"]}
    assert counts[chosen["id"]] == 1
    assert counts["A" if chosen["id"] == "B" else "B"] == 0
    other, _ = tenant(c, "other")
    assert c.get("/v1/experiments/exp1", headers={"X-API-Key": other}).status_code == 404


def test_turn_metric_labels_the_variant_without_dropping_the_unlabeled_series():
    _count({"type": "user_end_of_turn"})
    _count({"type": "user_end_of_turn"}, "A")
    text = METRICS.render()
    assert any(line.startswith("vox_turns_total ") for line in text.splitlines())
    assert 'vox_turns_total{variant="A"}' in text
