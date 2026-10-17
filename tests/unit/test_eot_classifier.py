import json

import numpy as np

from api.runtime import Runtime
from api.store import Store
from core.eot_classifier import DEFAULT_WEIGHTS, EotClassifier
from tests.unit.test_api import FakeTTS, FakeVAD, make, tenant


def test_silence_has_no_evidence_and_scores_even():
    clf = EotClassifier()
    assert clf.weights == DEFAULT_WEIGHTS
    assert clf.score(np.zeros(100, np.int16)) == 0.5
    assert clf.score(np.zeros(16000, np.int16)) == 0.5


def test_negative_filler_weight_scores_an_um_below_even():
    clf = EotClassifier({"bias": 0.0, "slope": 0.0, "decay": 0.0, "filler": -40.0, "voiced": 0.0})
    assert clf.score_features({"slope": 1.0, "decay": 0.5, "filler": 1, "voiced": 12}) < 0.5
    assert clf.score_features({"slope": 0.0, "decay": 0.0, "filler": 0, "voiced": 0}) == 0.5


def test_missing_model_file_keeps_the_hand_priors(tmp_path):
    missing = EotClassifier.load(str(tmp_path / "nope.json"))
    assert missing.weights == DEFAULT_WEIGHTS
    assert missing.score(np.zeros(80, np.int16)) == 0.5
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"bias": 0, "slope": 0, "decay": 0, "filler": -30, "voiced": 0}))
    loaded = EotClassifier.load(str(path))
    assert loaded.score_features({"filler": True, "slope": 3, "decay": 1, "voiced": 4}) < 0.5


def test_vox_eot_classifier_does_not_crash_when_the_model_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("VOX_EOT", "classifier")
    monkeypatch.setenv("VOX_EOT_MODEL", str(tmp_path / "absent.json"))
    rt = Runtime(object(), lambda v: None, lambda: None, Store())
    fn = rt._eot_sync()
    assert fn(np.zeros(64, np.int16), []) == 0.5


class _T:
    out_rate = 24000

    async def send_audio(self, pcm, rate):
        return None

    async def clear(self):
        return None

    async def end(self):
        return None


def test_agent_turn_values_land_on_the_session_and_out_of_range_is_rejected(monkeypatch):
    monkeypatch.setenv("VOX_END_MS", "700")
    monkeypatch.setenv("VOX_EOT_THRESHOLD", "0.55")
    monkeypatch.setenv("VOX_PATIENT_MS", "1200")
    c, store, _ = make()
    k, _ = tenant(c)
    h = {"X-API-Key": k}
    for bad in ({"endMs": 199}, {"endMs": 2001}, {"eotThreshold": 0.04}, {"eotThreshold": 1},
                {"patientMs": 399}, {"patientMs": 3001}, {"endMs": "slow"}):
        assert c.post("/v1/agents", json={"name": "a", "turn": bad}, headers=h).status_code == 422
    made = c.post("/v1/agents", json={"name": "a", "systemPrompt": "p",
                                      "turn": {"endMs": 200, "eotThreshold": 0.99, "patientMs": 400}}, headers=h)
    assert made.status_code == 200
    call = c.post("/v1/calls", json={"agentId": made.json()["id"]}, headers=h).json()
    sess = c.app.state.runtime.build_session(store.get_call_any(call["id"]), _T())
    assert sess.cfg.turn.end_ms == 200
    assert sess.cfg.turn.eot_threshold == 0.99
    assert sess.cfg.turn.patient_end_ms == 400
    # a later patch changes only the field it sends; the others stay, and env fills anything still unset
    patched = c.patch(f"/v1/agents/{made.json()['id']}", json={"turn": {"patientMs": 3000}}, headers=h)
    assert patched.status_code == 200
    partial = c.post("/v1/agents", json={"name": "b", "turn": {"patientMs": 1800}}, headers=h).json()
    call2 = c.post("/v1/calls", json={"agentId": partial["id"]}, headers=h).json()
    sess2 = c.app.state.runtime.build_session(store.get_call_any(call2["id"]), _T())
    assert sess2.cfg.turn.patient_end_ms == 1800
    assert sess2.cfg.turn.end_ms == 700 and sess2.cfg.turn.eot_threshold == 0.55
    bare = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=h).json()
    sess3 = c.app.state.runtime.build_session(store.get_call_any(bare["id"]), _T())
    assert (sess3.cfg.turn.end_ms, sess3.cfg.turn.eot_threshold, sess3.cfg.turn.patient_end_ms) == (700, 0.55, 1200)
    # the patched agent kept endMs and picked up the new patient wait
    call4 = c.post("/v1/calls", json={"agentId": made.json()["id"]}, headers=h).json()
    sess4 = c.app.state.runtime.build_session(store.get_call_any(call4["id"]), _T())
    assert sess4.cfg.turn.end_ms == 200 and sess4.cfg.turn.eot_threshold == 0.99
    assert sess4.cfg.turn.patient_end_ms == 3000


def test_runtime_classifier_ignores_history_and_uses_weights(monkeypatch, tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"bias": 0, "slope": 0, "decay": 0, "filler": -8, "voiced": 0}))
    monkeypatch.setenv("VOX_EOT", "classifier")
    monkeypatch.setenv("VOX_EOT_MODEL", str(path))
    rt = Runtime(object(), lambda voice: FakeTTS(), FakeVAD, Store())
    fn = rt._eot_sync()
    # silence still has no features, so the loaded filler weight never gets a chance to fire
    assert fn(np.zeros(100, np.int16), [{"role": "user", "content": "um"}]) == 0.5
