import numpy as np
import pytest
from scipy.signal import resample_poly

from core.audio import StreamResampler, mulaw_to_pcm16, pcm16_to_mulaw, resample


def ref_encode(sample: int) -> int:
    """Independent scalar G.711 mu-law encoder (classic table-free reference)."""
    BIAS, CLIP = 0x84, 32635
    sign = 0x80 if sample < 0 else 0
    s = min(abs(sample), CLIP) + BIAS
    i = (s >> 7) & 0xFF
    exponent = max(0, i.bit_length() - 1)
    mantissa = (s >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def ref_decode(code: int) -> int:
    code = ~code & 0xFF
    sign, exponent, mantissa = code & 0x80, (code >> 4) & 7, code & 0x0F
    s = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return -s if sign else s


def test_mulaw_anchor_values():
    enc = lambda v: pcm16_to_mulaw(np.array([v], dtype=np.int16))[0]
    assert enc(0) == 0xFF
    assert enc(-1) == 0x7F
    assert enc(32767) == 0x80
    assert enc(-32768) == 0x00


def test_mulaw_encode_matches_reference_for_every_int16():
    x = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    got = np.frombuffer(pcm16_to_mulaw(x), dtype=np.uint8)
    want = np.array([ref_encode(int(v)) for v in x], dtype=np.uint8)
    assert np.array_equal(got, want)


def test_mulaw_decode_matches_reference_for_every_code():
    codes = bytes(range(256))
    got = mulaw_to_pcm16(codes)
    want = np.array([ref_decode(c) for c in range(256)], dtype=np.int16)
    assert np.array_equal(got, want)


def test_mulaw_code_roundtrip_is_stable():
    codes = bytes(c for c in range(256) if c != 0x7F)  # 0x7F and 0xFF both decode to 0
    assert pcm16_to_mulaw(mulaw_to_pcm16(codes)) == codes


def test_mulaw_snr_on_speechlike_signal():
    t = np.arange(8000) / 8000
    x = (6000 * np.sin(2 * np.pi * 300 * t) + 2000 * np.sin(2 * np.pi * 1700 * t)).astype(np.int16)
    y = mulaw_to_pcm16(pcm16_to_mulaw(x)).astype(np.float64)
    snr = 10 * np.log10(np.sum(x.astype(np.float64) ** 2) / np.sum((x - y) ** 2))
    assert snr > 30


def _peak_hz(x, rate):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spec) * rate / len(x)


def test_oneshot_resample_preserves_tone_and_length():
    t = np.arange(16000) / 16000
    x = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    y = resample(x, 16000, 8000)
    assert len(y) == 8000
    assert abs(_peak_hz(y, 8000) - 440) < 2
    assert abs(np.max(np.abs(y)) - 8000) < 200


@pytest.mark.parametrize("src,dst", [(24000, 8000), (8000, 16000), (16000, 8000), (24000, 16000), (16000, 24000)])
def test_stream_resampler_matches_oneshot(src, dst):
    rng = np.random.default_rng(0)
    n = 24000 + 37  # deliberately not a multiple of the decimation factor
    t = np.arange(n) / src
    x = (5000 * np.sin(2 * np.pi * 523 * t) + 2500 * np.sin(2 * np.pi * 1111 * t)).astype(np.int16)

    from math import gcd
    g = gcd(src, dst)
    want = np.clip(np.rint(resample_poly(x.astype(np.float64), dst // g, src // g)), -32768, 32767).astype(np.int16)

    rs = StreamResampler(src, dst)
    out, i = [], 0
    while i < n:
        step = int(rng.integers(1, 700))
        out.append(rs.push(x[i : i + step]))
        i += step
    out.append(rs.flush())
    got = np.concatenate(out)

    assert len(got) == len(want)
    assert np.max(np.abs(got.astype(np.int32) - want.astype(np.int32))) <= 1


def test_stream_resampler_identity_passthrough():
    x = np.arange(100, dtype=np.int16)
    rs = StreamResampler(16000, 16000)
    assert np.array_equal(rs.push(x), x)
    assert len(rs.flush()) == 0
