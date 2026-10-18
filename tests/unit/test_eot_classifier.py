import json
import math

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


def test_default_weights_score_a_pitch_fall_above_a_pitch_rise():
    """Hand priors are applied to the same normalized cues as eot_prosody, not raw semitones/s."""
    clf = EotClassifier()
    fall = {"slope": -6.0, "decay": 0.75, "filler": 0, "voiced": 0}
    rise = {"slope": 6.0, "decay": 0.75, "filler": 0, "voiced": 0}
    assert clf.score_features(fall) > clf.score_features(rise)

    def hand(feats: dict) -> float:
        slope = float(feats.get("slope") or 0.0)
        decay = float(feats.get("decay") or 0.0)
        filler = 1.0 if feats.get("filler") else 0.0
        pitched = float(np.clip(-slope / 6.0, -1.0, 1.0))
        soft = float(np.clip((0.75 - decay) / 0.5, -1.0, 1.0))
        z = 0.3 + 1.5 * pitched + 0.9 * soft - 1.8 * filler
        return float(1.0 / (1.0 + np.exp(-z)))

    samples = (
        fall,
        rise,
        {"slope": 0.0, "decay": 0.25, "filler": 1, "voiced": 20},
        {"slope": -12.0, "decay": 1.5, "filler": 0, "voiced": 4},
    )
    for feats in samples:
        assert abs(clf.score_features(feats) - hand(feats)) < 1e-6


def test_a_weight_that_cannot_be_a_float_keeps_the_hand_priors(monkeypatch, tmp_path):
    path = tmp_path / "huge.json"
    path.write_text(json.dumps({"bias": 10 ** 400}))
    assert EotClassifier.load(str(path)).weights == DEFAULT_WEIGHTS
    monkeypatch.setenv("VOX_EOT", "classifier")
    monkeypatch.setenv("VOX_EOT_MODEL", str(path))
    rt = Runtime(object(), lambda v: None, lambda: None, Store())
    assert rt._eot_sync()(np.zeros(64, np.int16), []) == 0.5


def test_a_nonfinite_weight_keeps_the_hand_priors(tmp_path):
    """JSON 1e309 / Infinity / NaN become non-finite floats without raising, so they must not load."""
    from tests.unit.test_prosody import clip, tone

    hum = clip(tone(180, 180, 0.5), tone(180, 180, 0.6))
    filler = {"filler": True, "slope": 0.0, "decay": 0.75, "voiced": 0.0}
    prior = EotClassifier()
    prior_hum = prior.score(hum)
    prior_filler = prior.score_features(filler)
    assert math.isfinite(prior_hum) and prior_hum != 1.0
    assert math.isfinite(prior_filler) and prior_filler != 1.0
    # a finite sibling must not survive: one bad weight drops the whole file
    bodies = (
        '{"bias": 1e309, "filler": -40}',
        '{"bias": Infinity, "slope": 9}',
        '{"bias": -Infinity, "decay": 9}',
        '{"bias": NaN, "voiced": 9}',
    )
    for body in bodies:
        path = tmp_path / "bad.json"
        path.write_text(body)
        loaded = EotClassifier.load(str(path))
        assert loaded.weights == DEFAULT_WEIGHTS
        assert loaded.score(hum) == prior_hum
        assert loaded.score_features(filler) == prior_filler


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


def _fall_and_hum():
    from tests.unit.test_prosody import clip, tone

    fall = clip(tone(200, 200, 0.6), tone(200, 130, 0.4, 0.3, 0.08))
    hum = clip(tone(180, 180, 0.5), tone(180, 180, 0.6))
    return fall, hum


def test_classifier_scores_prosody_clips_and_a_loaded_bias_overrides_them(monkeypatch, tmp_path):
    """Hand priors match eot_prosody on real clips. A loaded finite bias must move the runtime score."""
    from core.prosody import eot_prosody

    fall, hum = _fall_and_hum()
    clf = EotClassifier()
    assert clf.score(fall) == eot_prosody(fall)
    assert clf.score(hum) == eot_prosody(hum)
    um = [{"role": "user", "content": "um"}]
    monkeypatch.setenv("VOX_EOT", "classifier")
    monkeypatch.delenv("VOX_EOT_MODEL", raising=False)
    rt = Runtime(object(), lambda v: None, lambda: None, Store())
    priors = rt._eot_sync()
    assert priors(fall, []) == eot_prosody(fall)
    assert priors(hum, um) == eot_prosody(hum)
    assert priors(fall, []) == priors(fall, um)
    assert priors(hum, []) == priors(hum, um)

    path = tmp_path / "bias.json"
    path.write_text(json.dumps({"bias": 20, "slope": 0, "decay": 0, "filler": 0, "voiced": 0}))
    monkeypatch.setenv("VOX_EOT_MODEL", str(path))
    loaded = rt._eot_sync()
    got_empty = loaded(fall, [])
    got_um = loaded(fall, um)
    assert got_empty == got_um
    assert got_empty > 0.99
    assert got_empty != eot_prosody(fall)


def test_published_turn_ends_and_unset_env_defaults(monkeypatch):
    for name in ("VOX_END_MS", "VOX_EOT_THRESHOLD", "VOX_PATIENT_MS"):
        monkeypatch.delenv(name, raising=False)
    c, store, _ = make()
    k, _ = tenant(c)
    h = {"X-API-Key": k}
    made = c.post("/v1/agents", json={"name": "edges", "systemPrompt": "p",
                                      "turn": {"endMs": 2000, "eotThreshold": 0.05}}, headers=h)
    assert made.status_code == 200
    call = c.post("/v1/calls", json={"agentId": made.json()["id"]}, headers=h).json()
    sess = c.app.state.runtime.build_session(store.get_call_any(call["id"]), _T())
    assert sess.cfg.turn.end_ms == 2000
    assert sess.cfg.turn.eot_threshold == 0.05
    bare = c.post("/v1/calls", json={"systemPrompt": "p"}, headers=h).json()
    bare_sess = c.app.state.runtime.build_session(store.get_call_any(bare["id"]), _T())
    assert (bare_sess.cfg.turn.end_ms, bare_sess.cfg.turn.eot_threshold,
            bare_sess.cfg.turn.patient_end_ms) == (640, 0.8, 1000)


def test_truncated_null_and_non_numeric_model_files_keep_the_hand_priors(monkeypatch, tmp_path):
    from core.prosody import eot_prosody

    fall, _hum = _fall_and_hum()
    prior = eot_prosody(fall)
    bodies = (
        '{"bias": 0.1, "slope":',
        "null",
        json.dumps({"bias": "nope", "slope": 9}),
    )
    monkeypatch.setenv("VOX_EOT", "classifier")
    rt = Runtime(object(), lambda v: None, lambda: None, Store())
    for body in bodies:
        path = tmp_path / "bad.json"
        path.write_text(body)
        assert EotClassifier.load(str(path)).weights == DEFAULT_WEIGHTS
        monkeypatch.setenv("VOX_EOT_MODEL", str(path))
        assert rt._eot_sync()(fall, []) == prior
