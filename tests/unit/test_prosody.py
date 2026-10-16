import numpy as np

from core.prosody import eot_prosody, features

SR = 16000


def tone(f0a, f0b, dur, a0=0.3, a1=0.3):
    n = int(SR * dur)
    ph = 2 * np.pi * np.cumsum(np.linspace(f0a, f0b, n)) / SR
    return sum(np.sin(h * ph) / h for h in (1, 2, 3)) * np.linspace(a0, a1, n)


def clip(*segs, tail=0.3):
    x = np.concatenate([np.zeros(3200)] + list(segs) + [np.zeros(int(SR * tail))])
    return (x * 32767 / 2).astype(np.int16)


def test_falling_softening_ending_scores_as_finished():
    assert eot_prosody(clip(tone(200, 200, 0.6), tone(200, 130, 0.4, 0.3, 0.08))) > 0.8


def test_steady_hum_is_a_filler_and_scores_as_unfinished():
    a = clip(tone(180, 180, 0.5), tone(180, 180, 0.6))
    assert features(a)["filler"] and eot_prosody(a) < 0.3


def test_rising_full_volume_ending_scores_as_unfinished():
    assert eot_prosody(clip(tone(160, 160, 0.5), tone(160, 230, 0.4))) < 0.35


def test_too_little_speech_or_silence_means_no_evidence():
    assert 0.4 < eot_prosody(clip(tone(200, 200, 0.1))) < 0.6
    assert eot_prosody(np.zeros(16000, np.int16)) == 0.5
    assert eot_prosody(np.zeros(100, np.int16)) == 0.5


def test_scores_are_probabilities_and_ignore_trailing_silence_length():
    x1 = clip(tone(200, 200, 0.6), tone(200, 130, 0.4, 0.3, 0.08), tail=0.1)
    x2 = clip(tone(200, 200, 0.6), tone(200, 130, 0.4, 0.3, 0.08), tail=0.6)
    assert 0 <= eot_prosody(x1) <= 1 and abs(eot_prosody(x1) - eot_prosody(x2)) < 0.05
