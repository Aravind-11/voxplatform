"""Model and voice A/B: stable assignment, agent-save validation, per-call model wrapper."""
import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

from api import security
from api.app import Settings, create_app
from api.experiments import ExperimentError, choose_variant, validate_experiment
from api.runtime import Runtime, _count
from api.store import Store
from core.metrics import METRICS
from tests.unit.test_api import FakeLLM, FakeTTS, FakeVAD, make, sign, tenant
from tests.unit.test_recovery import resume

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


def test_variant_write_keeps_caller_identity_stored_after_the_call_object_was_loaded():
    """SIP (and outbound) write from/to/callSid onto the row after the in-memory call was loaded.
    Choosing a variant must not replace that metadata with only the experiment fields."""
    c, store, _ = make()
    key, tid = tenant(c)
    h = {"X-API-Key": key}
    exp = {"id": "exp1", "variants": [{"id": "A", "weight": 50, "voice": "af_heart"},
                                      {"id": "B", "weight": 50, "voice": "af_bella"}]}
    agent = c.post("/v1/agents", json={"name": "n", "systemPrompt": "Caller {{caller}} dialled {{callee}}.",
                                       "experiment": exp}, headers=h)
    assert agent.status_code == 200
    created = c.post("/v1/calls", json={"agentId": agent.json()["id"]}, headers=h).json()
    call = store.get_call_any(created["id"])
    assert call["metadata"] == {}
    store.update_call(call["id"], metadata={"from": "+15551110000", "to": "+15552220000", "callSid": "CA123"})
    assert call["metadata"] == {}
    rt = Runtime(SharedLLM(), lambda voice: FakeTTS(), FakeVAD, store)
    sess = rt.build_session(call, T())
    stored = store.get_call(tid, call["id"])["metadata"]
    assert stored["from"] == "+15551110000"
    assert stored["to"] == "+15552220000"
    assert stored["callSid"] == "CA123"
    assert stored["experiment"] == "exp1"
    assert stored["variant"] in ("A", "B")
    assert sess.history[0]["content"] == "Caller +15551110000 dialled +15552220000."


def _until(store, tid, cfg, meta, pred):
    for _ in range(40):
        call = store.create_call(tid, None, "twilio", cfg, meta)
        if pred(call):
            return call
    raise AssertionError("no call id landed in the other bucket")


async def test_continuation_keeps_the_root_variant_and_a_client_variant_does_not_pin():
    """A resume is a new call row. Its bucket is the root call, not the new id and not metadata.variant."""
    c, store, _ = make()
    key, tid = tenant(c)
    h = {"X-API-Key": key}
    other = store.create_tenant("other")
    saved = c.post("/v1/agents", json={"name": "n", "systemPrompt": "base prompt", "voice": "af_heart",
                                       "experiment": EXP}, headers=h)
    assert saved.status_code == 200
    cfg = saved.json()["config"]
    exp = cfg["experiment"]
    llm = SharedLLM()
    rt = Runtime(llm, lambda voice: FakeTTS(), FakeVAD, store)

    pinned = _until(store, tid, cfg, {"variant": "A", "experiment": "exp1"},
                    lambda call: choose_variant(exp, call["id"])["id"] != "A")
    rt.build_session(pinned, T())
    assert store.get_call_any(pinned["id"])["metadata"]["variant"] != "A"

    root = store.create_call(tid, saved.json()["id"], "twilio", cfg, {"from": "+1555", "to": "+1666"})
    root_vid = choose_variant(exp, root["id"])["id"]
    child = _until(store, tid, cfg, {"resumedFrom": root["id"], "from": "+1555", "variant": "A" if root_vid == "B" else "B"},
                   lambda call: choose_variant(exp, call["id"])["id"] != root_vid)
    sess = rt.build_session(child, T())
    assert store.get_call_any(child["id"])["metadata"]["variant"] == root_vid
    prompt = next(v.get("systemPrompt") for v in exp["variants"] if v["id"] == root_vid) or "base prompt"
    assert sess.history[0]["content"] == prompt
    async for _ in sess.llm.stream([], np.zeros(0, np.int16), [], []):
        pass
    assert llm.model == "base-model"
    assert llm.seen[-1] == (("ultravox" if root_vid == "A" else "ultravox-b"), "shared")

    # The id recorded on the ancestor sticks after weights would rebucket the root.
    rt.build_session(root, T())
    parent = store.get_call_any(root["id"])
    assert parent["metadata"]["variant"] == root_vid
    flipped = "B" if root_vid == "A" else "A"
    store.update_call(root["id"], metadata={**parent["metadata"], "variant": flipped})
    stuck = store.create_call(tid, None, "twilio", cfg, {"resumedFrom": root["id"], "variant": root_vid})
    rt.build_session(stuck, T())
    assert store.get_call_any(stuck["id"])["metadata"]["variant"] == flipped

    only_a = {"id": "exp1", "variants": [{"id": "A", "weight": 1, "voice": "af_heart", "systemPrompt": "Only A."}]}
    store.update_call(root["id"], metadata={**store.get_call_any(root["id"])["metadata"], "variant": "B"})
    stale = store.create_call(tid, None, "twilio", {**cfg, "systemPrompt": "base prompt", "experiment": only_a},
                              {"resumedFrom": root["id"], "variant": "B"})
    stale_sess = rt.build_session(stale, T())
    assert store.get_call_any(stale["id"])["metadata"]["variant"] == "A"
    assert stale_sess.history[0]["content"] == "Only A."

    foreign = store.create_call(other, None, "twilio", cfg, {"experiment": "exp1", "variant": "B"})
    local = _until(store, tid, cfg, {"resumedFrom": foreign["id"], "variant": "B"},
                   lambda call: choose_variant(exp, call["id"])["id"] != "B")
    rt.build_session(local, T())
    assert store.get_call_any(local["id"])["metadata"]["variant"] == choose_variant(exp, local["id"])["id"]

    # Same cap as _earlier_messages: 4 rows including this call. The call 4 hops up is not an ancestor.
    c0 = store.create_call(tid, None, "twilio", cfg, {})
    c1 = store.create_call(tid, None, "twilio", cfg, {"resumedFrom": c0["id"]})
    c2 = store.create_call(tid, None, "twilio", cfg, {"resumedFrom": c1["id"]})
    c3 = store.create_call(tid, None, "twilio", cfg, {"resumedFrom": c2["id"]})
    want = choose_variant(exp, c1["id"])["id"]
    pinned_id = "A" if want == "B" else "B"
    store.update_call(c0["id"], metadata={"experiment": "exp1", "variant": pinned_id})
    far = _until(store, tid, cfg, {"resumedFrom": c3["id"]},
                 lambda call: choose_variant(exp, call["id"])["id"] != want)
    rt.build_session(far, T())
    assert store.get_call_any(far["id"])["metadata"]["variant"] == want
    store.update_call(c1["id"], metadata={"resumedFrom": c0["id"], "experiment": "exp1", "variant": pinned_id})
    near = _until(store, tid, cfg, {"resumedFrom": c1["id"]},
                  lambda call: choose_variant(exp, call["id"])["id"] != pinned_id)
    rt.build_session(near, T())
    assert store.get_call_any(near["id"])["metadata"]["variant"] == pinned_id

    # Twilio resume creates the continuation row. It must keep the root variant, including on the report.
    pair = None
    for _ in range(20):
        call = store.create_call(tid, saved.json()["id"], "twilio", cfg,
                                 {"from": "+15550001", "to": "+15550002", "callSid": "CA1"})
        vid = choose_variant(exp, call["id"])["id"]
        assert "<Stream" in resume(c, call["id"]).text
        nxt = [x for x in store.list_calls(tid, 1000) if x["metadata"].get("resumedFrom") == call["id"]][0]
        if choose_variant(nxt["config"]["experiment"], nxt["id"])["id"] != vid:
            pair = (call, nxt, vid)
            break
    assert pair is not None
    call, nxt, vid = pair
    before = {row["id"]: row["calls"] for row in c.get("/v1/experiments/exp1", headers=h).json()["variants"]}
    rt.build_session(store.get_call_any(call["id"]), T())
    rt.build_session(nxt, T())
    assert store.get_call_any(nxt["id"])["metadata"]["variant"] == vid
    assert store.get_call_any(call["id"])["metadata"]["variant"] == vid
    assert store.get_call_any(nxt["id"])["metadata"]["from"] == "+15550001"
    after = {row["id"]: row["calls"] for row in c.get("/v1/experiments/exp1", headers=h).json()["variants"]}
    other_vid = "A" if vid == "B" else "B"
    assert after[vid] == before.get(vid, 0) + 2
    assert after[other_vid] == before.get(other_vid, 0)


async def test_failed_transfer_keeps_the_variant_prompt_and_one_transfer_note():
    c, store, _ = make()
    key, tid = tenant(c)
    h = {"X-API-Key": key}
    exp = {"id": "exp1", "variants": [
        {"id": "A", "weight": 1, "voice": "af_heart", "systemPrompt": "PROMPT-A", "model": "model-a"},
        {"id": "B", "weight": 1, "voice": "af_bella", "systemPrompt": "PROMPT-B", "model": "model-b"}]}
    agent = c.post("/v1/agents", json={"name": "a", "systemPrompt": "Be brief.", "greeting": "Hello!",
                                       "experiment": exp}, headers=h)
    assert agent.status_code == 200
    cfg = agent.json()["config"]
    note = "\nThe caller asked to be transferred to billing, who did not answer. Offer to help or to take a message."
    found = None
    for _ in range(30):
        call = store.create_call(tid, agent.json()["id"], "twilio", cfg, {"from": "+1555", "to": "+1666"})
        path, q = f"/twilio/transfer-result/{call['id']}", "to=billing"
        miss = {"DialCallStatus": "no-answer"}
        r = c.post(f"{path}?{q}", data=miss, headers=sign(f"https://vox.example.com{path}?{q}", miss))
        assert r.status_code == 200 and "<Connect>" in r.text
        child = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == call["id"]][0]
        root_vid = choose_variant(call["config"]["experiment"], call["id"])["id"]
        if choose_variant(child["config"]["experiment"], child["id"])["id"] != root_vid:
            found = (call, child, root_vid)
            break
    assert found is not None
    call, child, root_vid = found
    assert "Be brief." in child["config"]["systemPrompt"] and child["config"]["systemPrompt"].count("did not answer") == 1
    assert child["metadata"]["promptSuffix"] == note
    llm = SharedLLM()
    voices = []
    rt = Runtime(llm, lambda voice: voices.append(voice) or FakeTTS(), FakeVAD, store)
    sess = rt.build_session(child, T())
    assert sess.history[0]["content"] == ("PROMPT-A" if root_vid == "A" else "PROMPT-B") + note
    assert voices == ["af_heart" if root_vid == "A" else "af_bella"]
    assert store.get_call_any(child["id"])["metadata"]["variant"] == root_vid
    assert store.get_call_any(child["id"])["metadata"]["promptSuffix"] == note
    async for _ in sess.llm.stream([], np.zeros(0, np.int16), [], []):
        pass
    assert llm.model == "base-model"
    assert llm.seen[-1] == (("model-a" if root_vid == "A" else "model-b"), "shared")
    again = store.create_call(tid, agent.json()["id"], "twilio", child["config"],
                              {**store.get_call_any(child["id"])["metadata"], "resumedFrom": child["id"], "resumes": 1})
    assert rt.build_session(again, T()).history[0]["content"] == ("PROMPT-A" if root_vid == "A" else "PROMPT-B") + note
    assert store.get_call_any(again["id"])["metadata"]["variant"] == root_vid

    voice_only = {"id": "exp-voice", "variants": [{"id": "A", "weight": 1, "voice": "af_bella"}]}
    agent2 = c.post("/v1/agents", json={"name": "b", "systemPrompt": "Be brief.", "experiment": voice_only}, headers=h)
    assert agent2.status_code == 200
    call2 = store.create_call(tid, agent2.json()["id"], "twilio", agent2.json()["config"], {})
    path, q = f"/twilio/transfer-result/{call2['id']}", "to=billing"
    miss = {"DialCallStatus": "busy"}
    assert "<Connect>" in c.post(f"{path}?{q}", data=miss, headers=sign(f"https://vox.example.com{path}?{q}", miss)).text
    child2 = [x for x in store.list_calls(tid) if x["metadata"].get("resumedFrom") == call2["id"]][0]
    assert rt.build_session(child2, T()).history[0]["content"] == "Be brief." + note


def test_variant_prompt_placeholders_are_required_on_api_calls_and_outbound():
    c, store, _ = make()
    key, _ = tenant(c)
    h = {"X-API-Key": key}
    exp = {"id": "exp1", "variants": [{"id": "A", "weight": 1, "systemPrompt": "Hello {{name}}."}]}
    agent = c.post("/v1/agents", json={"name": "n", "systemPrompt": "Base.", "experiment": exp}, headers=h)
    assert agent.status_code == 200
    missing = c.post("/v1/calls", json={"agentId": agent.json()["id"]}, headers=h)
    assert missing.status_code == 422 and "name" in missing.json()["detail"] and "variant" in missing.json()["detail"]
    builtin = {"id": "exp1", "variants": [{"id": "A", "weight": 1, "systemPrompt": "Hi {{caller}}."}]}
    caller_agent = c.post("/v1/agents", json={"name": "c", "systemPrompt": "Base.", "experiment": builtin}, headers=h).json()
    assert c.post("/v1/calls", json={"agentId": caller_agent["id"]}, headers=h).status_code == 200
    both = {"id": "exp1", "variants": [{"id": "A", "weight": 1, "systemPrompt": "Call {{customer}} and {{extra}}."}]}
    mixed = c.post("/v1/agents", json={"name": "m", "systemPrompt": "Call {{customer}}.", "experiment": both,
                                       "variables": {"customer": "Pat"}}, headers=h)
    assert mixed.status_code == 200
    only_customer = c.post("/v1/calls", json={"agentId": mixed.json()["id"], "variables": {"customer": "Dana"}}, headers=h)
    assert only_customer.status_code == 422 and "extra" in only_customer.json()["detail"]
    ok = c.post("/v1/calls", json={"agentId": agent.json()["id"], "variables": {"name": "Ada"}}, headers=h)
    assert ok.status_code == 200
    filled = c.post("/v1/calls", json={"agentId": mixed.json()["id"],
                                       "variables": {"customer": "Dana", "extra": "the plan"}}, headers=h)
    assert filled.status_code == 200
    rt = Runtime(SharedLLM(), lambda voice: FakeTTS(), FakeVAD, store)
    assert rt.build_session(store.get_call_any(ok.json()["id"]), T()).history[0]["content"] == "Hello Ada."
    assert rt.build_session(store.get_call_any(filled.json()["id"]), T()).history[0]["content"] == "Call Dana and the plan."

    def handler(req):
        handler.called = True
        return httpx.Response(201, json={"sid": "CA1"})
    handler.called = False
    store2 = Store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt2 = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store2, http=http)
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        twilio_account_sid="ACtest", twilio_number="+15550001111")
    c2 = TestClient(create_app(store2, rt2, security.ConcurrencyLimiter(2), None, settings))
    key2, _ = tenant(c2)
    h2 = {"X-API-Key": key2}
    agent2 = c2.post("/v1/agents", json={"name": "n", "systemPrompt": "Base.", "experiment": exp}, headers=h2)
    assert agent2.status_code == 200
    denied = c2.post("/v1/calls/outbound", json={"agentId": agent2.json()["id"], "to": "+15552223333"}, headers=h2)
    assert denied.status_code == 422 and "name" in denied.json()["detail"] and "variant" in denied.json()["detail"]
    assert handler.called is False
    placed = c2.post("/v1/calls/outbound", json={"agentId": agent2.json()["id"], "to": "+15552223333",
                                                 "variables": {"name": "Ada"}}, headers=h2)
    assert placed.status_code == 200 and handler.called is True


def test_non_finite_weights_are_rejected_and_do_not_collapse_the_bucket():
    with pytest.raises(ExperimentError):
        validate_experiment({"id": "e", "variants": [{"id": "A", "weight": 10**400}, {"id": "B", "weight": 1}]})
    with pytest.raises(ExperimentError):
        validate_experiment({"id": "e", "variants": [{"id": "A", "weight": 1e308}, {"id": "B", "weight": 1}]})
    with pytest.raises(ExperimentError):
        validate_experiment({"id": "e", "variants": [{"id": "A", "weight": 1e308}, {"id": "B", "weight": 1e308}]})
    with pytest.raises(ExperimentError):
        validate_experiment({"id": "e", "variants": [{"id": "A", "weight": float("inf")}]})
    with pytest.raises(ExperimentError):
        validate_experiment({"id": "e", "variants": [{"id": "A", "weight": float("nan")}]})
    with pytest.raises(ExperimentError):
        validate_experiment({"id": "e", "variants": [{"id": "A", "weight": True}]})
    one = validate_experiment({"id": "e", "variants": [{"id": "A", "weight": 1e308}]})
    assert choose_variant(one, "call")["id"] == "A"
    kept = validate_experiment({"id": "e", "variants": [{"id": "A", "weight": 2}, {"id": "B", "weight": 1.5}]})
    assert kept["variants"][0]["weight"] == 2 and kept["variants"][1]["weight"] == 1.5
    for bad in (
        {"id": "e", "variants": [{"id": "A", "weight": 1e308}, {"id": "B", "weight": 1e308}]},
        {"id": "e", "variants": [{"id": "A", "weight": 1e308}, {"id": "B", "weight": 1}]},
        {"id": "e", "variants": [{"id": "A", "weight": 10**400}, {"id": "B", "weight": 1}]},
    ):
        with pytest.raises(ExperimentError):
            choose_variant(bad, "call_1")
    c, store, _ = make()
    key, _ = tenant(c)
    h = {"X-API-Key": key}
    saved = c.post("/v1/agents", json={"name": "n", "experiment": EXP}, headers=h)
    assert saved.status_code == 200
    aid = saved.json()["id"]
    for weight_a, weight_b in ((10**400, 1), (1e308, 1), (1e308, 1e308)):
        body = {"id": "exp1", "variants": [{"id": "A", "weight": weight_a}, {"id": "B", "weight": weight_b}]}
        assert c.post("/v1/agents", json={"name": "n", "experiment": body}, headers=h).status_code == 422
        patched = c.patch(f"/v1/agents/{aid}", json={"experiment": body}, headers=h)
        assert patched.status_code == 422
        assert store.get_agent_any(aid)["config"]["experiment"]["variants"][0]["weight"] == 50


def test_blank_variant_prompt_keeps_the_agent_prompt_and_blank_model_does_not_wrap():
    """A direct call config can carry a blank variant prompt that validation would have dropped."""
    _, store, _ = make()
    tid = store.create_tenant("t")
    llm = SharedLLM()
    rt = Runtime(llm, lambda voice: FakeTTS(), FakeVAD, store)
    for blank in ("", "   "):
        cfg = {"systemPrompt": "agent prompt", "voice": "af_heart",
               "experiment": {"id": "exp1", "variants": [
                   {"id": "A", "weight": 1, "voice": "af_bella", "systemPrompt": blank, "model": "   "}]}}
        call = store.create_call(tid, None, "webSocket", cfg, {})
        voices = []
        rt.make_tts = lambda voice: voices.append(voice) or FakeTTS()
        sess = rt.build_session(call, T())
        assert sess.history[0]["content"] == "agent prompt"
        assert sess.llm is llm
        assert voices == ["af_bella"]
        voices.clear()


def test_turn_metric_labels_the_variant_without_dropping_the_unlabeled_series():
    _count({"type": "user_end_of_turn"})
    _count({"type": "user_end_of_turn"}, "A")
    text = METRICS.render()
    assert any(line.startswith("vox_turns_total ") for line in text.splitlines())
    assert 'vox_turns_total{variant="A"}' in text


def _transfer_note(who):
    return f"\nThe caller asked to be transferred to {who}, who did not answer. Offer to help or to take a message."


def _fail_transfer(c, call_id, to):
    path, q = f"/twilio/transfer-result/{call_id}", f"to={to}"
    form = {"DialCallStatus": "no-answer"}
    r = c.post(f"{path}?{q}", data=form, headers=sign(f"https://vox.example.com{path}?{q}", form))
    assert r.status_code == 200 and "<Connect>" in r.text, r.text
    return r


def _resumed_from(store, tid, parent_id):
    rows = [x for x in store.list_calls(tid, 100) if (x.get("metadata") or {}).get("resumedFrom") == parent_id]
    assert len(rows) == 1
    return rows[0]


def test_second_failed_transfer_keeps_every_note_and_a_voice_only_variant_does_not_repeat_them():
    """A second failed transfer must not drop the first note when the variant prompt replaces the agent prompt."""
    c, store, _ = make()
    key, tid = tenant(c)
    h = {"X-API-Key": key}
    billing, sales = _transfer_note("billing"), _transfer_note("sales")
    exp = {"id": "exp1", "variants": [
        {"id": "A", "weight": 1, "voice": "af_heart", "systemPrompt": "PROMPT-A", "model": "model-a"}]}
    agent = c.post("/v1/agents", json={"name": "a", "systemPrompt": "Be brief.", "greeting": "Hello!",
                                       "experiment": exp}, headers=h)
    assert agent.status_code == 200
    call = store.create_call(tid, agent.json()["id"], "twilio", agent.json()["config"],
                             {"from": "+1555", "to": "+1666"})
    _fail_transfer(c, call["id"], "billing")
    child = _resumed_from(store, tid, call["id"])
    _fail_transfer(c, child["id"], "sales")
    grand = _resumed_from(store, tid, child["id"])
    assert grand["config"]["systemPrompt"] == "Be brief." + billing + sales
    assert grand["metadata"]["promptSuffix"] == billing + sales
    assert grand["metadata"]["from"] == "+1555"
    rt = Runtime(FakeLLM(), lambda voice: FakeTTS(), FakeVAD, store)
    assert rt.build_session(grand, T()).history[0]["content"] == "PROMPT-A" + billing + sales

    voice_only = {"id": "exp-voice", "variants": [{"id": "A", "weight": 1, "voice": "af_bella"}]}
    agent2 = c.post("/v1/agents", json={"name": "b", "systemPrompt": "Be brief.", "experiment": voice_only}, headers=h)
    assert agent2.status_code == 200
    call2 = store.create_call(tid, agent2.json()["id"], "twilio", agent2.json()["config"], {"from": "+1555"})
    _fail_transfer(c, call2["id"], "billing")
    child2 = _resumed_from(store, tid, call2["id"])
    _fail_transfer(c, child2["id"], "sales")
    grand2 = _resumed_from(store, tid, child2["id"])
    heard = rt.build_session(grand2, T()).history[0]["content"]
    assert heard == "Be brief." + billing + sales
    assert heard.count("did not answer") == 2


def test_client_prompt_suffix_is_not_rendered_unless_the_configured_prompt_ends_with_it():
    """metadata.promptSuffix is a transfer addendum only when cfg systemPrompt already ends with that text."""
    c, store, _ = make()
    key, tid = tenant(c)
    h = {"X-API-Key": key}
    injected = "\nINJECTED by client"
    exp = {"id": "exp1", "variants": [{"id": "A", "weight": 1, "systemPrompt": "VARIANT", "voice": "af_heart"}]}
    agent = c.post("/v1/agents", json={"name": "n", "systemPrompt": "BASE", "experiment": exp}, headers=h)
    assert agent.status_code == 200
    aid = agent.json()["id"]
    created = c.post("/v1/calls", json={"agentId": aid, "systemPrompt": "OVERRIDE",
                                        "metadata": {"promptSuffix": injected, "from": "+1555"}}, headers=h)
    assert created.status_code == 200
    rt = Runtime(FakeLLM(), lambda voice: FakeTTS(), FakeVAD, store)
    call = store.get_call_any(created.json()["id"])
    assert rt.build_session(call, T()).history[0]["content"] == "VARIANT"
    _fail_transfer(c, call["id"], "billing")
    child = _resumed_from(store, tid, call["id"])
    note = _transfer_note("billing")
    assert child["metadata"]["promptSuffix"] == note
    assert injected not in child["config"]["systemPrompt"]
    assert rt.build_session(child, T()).history[0]["content"] == "VARIANT" + note

    blank = c.post("/v1/calls", json={"agentId": aid, "metadata": {"promptSuffix": "   "}}, headers=h)
    assert blank.status_code == 200
    assert rt.build_session(store.get_call_any(blank.json()["id"]), T()).history[0]["content"] == "VARIANT"
    tail = store.create_call(tid, aid, "webSocket",
                             {"systemPrompt": "BASE" + note, "experiment": agent.json()["config"]["experiment"]},
                             {"promptSuffix": note})
    assert rt.build_session(tail, T()).history[0]["content"] == "VARIANT" + note

    def handler(req):
        return httpx.Response(201, json={"sid": "CA1"})
    store2 = Store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rt2 = Runtime(FakeLLM(), lambda v: FakeTTS(), FakeVAD, store2, http=http)
    settings = Settings(admin_key="adm", public_base_url="https://vox.example.com", twilio_auth_token="tok",
                        twilio_account_sid="ACtest", twilio_number="+15550001111")
    c2 = TestClient(create_app(store2, rt2, security.ConcurrencyLimiter(2), None, settings))
    key2, _ = tenant(c2)
    h2 = {"X-API-Key": key2}
    agent2 = c2.post("/v1/agents", json={"name": "n", "systemPrompt": "BASE", "experiment": exp}, headers=h2)
    assert agent2.status_code == 200
    placed = c2.post("/v1/calls/outbound", json={"agentId": agent2.json()["id"], "to": "+15552223333",
                                                 "metadata": {"promptSuffix": injected}}, headers=h2)
    assert placed.status_code == 200
    outbound = store2.get_call_any(placed.json()["id"])
    assert rt2.build_session(outbound, T()).history[0]["content"] == "VARIANT"


def test_variant_voice_must_speak_the_agent_language_on_create_and_on_a_language_only_patch(monkeypatch):
    monkeypatch.delenv("VOX_NO_ESPEAK", raising=False)
    c, store, _ = make()
    key, tid = tenant(c)
    h = {"X-API-Key": key}
    english = {"id": "exp1", "variants": [
        {"id": "A", "weight": 1, "voice": "af_heart"},
        {"id": "B", "weight": 1, "voice": "af_bella"}]}
    spanish = {"id": "exp-es", "variants": [
        {"id": "A", "weight": 1, "voice": "ef_dora"},
        {"id": "B", "weight": 1, "voice": "em_alex"}]}
    before = len(store.list_agents(tid))
    mismatch = c.post("/v1/agents", json={"name": "n", "language": "es", "experiment": english}, headers=h)
    assert mismatch.status_code == 422
    assert "af_heart" in mismatch.json()["detail"] and "speaks en" in mismatch.json()["detail"]
    inverse = c.post("/v1/agents", json={"name": "n", "language": "en", "voice": "af_heart", "experiment": {
        "id": "exp1", "variants": [{"id": "A", "weight": 1, "voice": "ef_dora"}]}}, headers=h)
    assert inverse.status_code == 422 and "speaks es" in inverse.json()["detail"] and "ef_dora" in inverse.json()["detail"]
    unknown = c.post("/v1/agents", json={"name": "n", "language": "es", "experiment": {
        "id": "exp1", "variants": [{"id": "A", "weight": 1, "voice": "not_a_voice"}]}}, headers=h)
    assert unknown.status_code == 422 and "unknown voice" in unknown.json()["detail"]
    assert len(store.list_agents(tid)) == before

    saved = c.post("/v1/agents", json={"name": "n", "language": "es", "experiment": spanish}, headers=h)
    assert saved.status_code == 200
    cfg = saved.json()["config"]
    assert cfg["language"] == "es" and cfg["voice"] == "ef_dora"
    assert [v["voice"] for v in cfg["experiment"]["variants"]] == ["ef_dora", "em_alex"]
    aid = saved.json()["id"]
    keep = store.get_agent_any(aid)
    swapped = c.patch(f"/v1/agents/{aid}", json={"experiment": english}, headers=h)
    assert swapped.status_code == 422 and "speaks en" in swapped.json()["detail"]
    assert store.get_agent_any(aid)["config"] == keep["config"]

    en = c.post("/v1/agents", json={"name": "en", "systemPrompt": "BASE", "experiment": english}, headers=h)
    assert en.status_code == 200
    eid = en.json()["id"]
    frozen = store.get_agent_any(eid)
    lang = c.patch(f"/v1/agents/{eid}", json={"language": "es"}, headers=h)
    assert lang.status_code == 422 and "speaks en" in lang.json()["detail"] and "af_heart" in lang.json()["detail"]
    assert store.get_agent_any(eid)["config"] == frozen["config"]
    assert store.get_agent_any(eid)["config"].get("language") in (None, "en")
    assert store.get_agent_any(eid)["config"]["experiment"]["variants"][0]["voice"] == "af_heart"

    greeted = c.patch(f"/v1/agents/{aid}", json={"greeting": "Hola"}, headers=h)
    assert greeted.status_code == 200 and greeted.json()["config"]["language"] == "es"
    assert greeted.json()["config"]["experiment"]["variants"][0]["voice"] == "ef_dora"
    snap = store.get_agent_any(aid)
    back = c.patch(f"/v1/agents/{aid}", json={"language": "en"}, headers=h)
    assert back.status_code == 422 and "speaks es" in back.json()["detail"]
    assert store.get_agent_any(aid)["config"] == snap["config"]

    cleared = c.patch(f"/v1/agents/{aid}", json={"experiment": None, "language": "en"}, headers=h)
    assert cleared.status_code == 200
    assert "experiment" not in cleared.json()["config"]
    assert cleared.json()["config"]["language"] == "en" and cleared.json()["config"]["voice"] == "af_heart"
