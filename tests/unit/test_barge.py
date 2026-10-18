import asyncio

from api.runtime import Runtime
from api.store import Store
from core.barge import PlaybackTracker, trim_history
from core.session import SessionConfig, ToolCall, ToolResult
from tests.unit.test_api import FakeLLM as ApiLLM, FakeTTS as ApiTTS, FakeVAD
from tests.unit.test_session import FakeLLM, FakeTTS, Rig


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make():
    clk = Clock()
    tr = PlaybackTracker(clk)
    tr.on_sent("Hello there friend.", 1000)          # plays 0..1000
    tr.on_sent("How can I help you today?", 2000)    # queued: plays 1000..3000
    return clk, tr


def test_partial_first_chunk():
    clk, tr = make()
    clk.t = 500
    assert tr.heard() == ("Hello", True)


def test_boundary_between_chunks_counts_as_cut():
    clk, tr = make()
    clk.t = 1000
    assert tr.heard() == ("Hello there friend.", True)


def test_partial_second_chunk():
    clk, tr = make()
    clk.t = 2000
    assert tr.heard() == ("Hello there friend. How can I", True)


def test_everything_played_is_not_a_cut():
    clk, tr = make()
    clk.t = 3500
    assert tr.heard() == ("Hello there friend. How can I help you today?", False)
    assert not tr.is_playing()


def test_nothing_heard_yet_when_output_latency_not_elapsed():
    clk = Clock()
    tr = PlaybackTracker(clk, output_latency_ms=200)
    tr.on_sent("Hello there friend.", 1000)
    clk.t = 150
    assert tr.heard() == ("", True)
    clk.t = 700
    text, cut = tr.heard()
    assert text == "Hello" and cut


def test_back_to_back_chunks_do_not_overlap_when_sent_late():
    clk = Clock()
    tr = PlaybackTracker(clk)
    tr.on_sent("One two.", 500)       # 0..500
    clk.t = 2000                      # long gap before the next chunk is sent
    tr.on_sent("Three four.", 500)    # 2000..2500, not 500..1000
    clk.t = 1000
    assert tr.heard() == ("One two.", True)


def test_trim_history_variants():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "full long answer the caller never finished hearing"}]
    cut = trim_history(msgs, "full long", True)
    assert cut[-1]["content"] == "full long [interrupted]"
    assert msgs[-1]["content"].startswith("full long answer")  # original untouched
    assert trim_history(msgs, "all of it", False)[-1]["content"] == "all of it"
    assert trim_history(msgs, "", True)[-1]["role"] == "user"


def test_acked_mark_limits_what_was_heard_even_if_the_clock_is_ahead():
    clk = Clock()
    # Give-up window is longer than this jump: a clock far past the audio still does not
    # count an unacked chunk as heard, and the unacked duration stays buffered.
    tr = PlaybackTracker(clk, mark_timeout_ms=60_000)
    tr.on_sent("Hello there friend.", 1000)
    tr.arm_mark("m1")
    tr.on_sent("How can I help you today?", 2000)
    tr.arm_mark("m2")
    clk.t = 10_000
    # nothing acked yet: the clock estimate still applies
    assert tr.heard() == ("Hello there friend. How can I help you today?", False)
    tr.note_mark("m1")
    text, cut = tr.heard()
    assert text == "Hello there friend." and cut
    assert tr.is_playing() and tr.buffered_ms() == 2000
    assert tr.acked_through == 0 and tr.pending_marks == {"m2": 1}
    tr.note_mark("nope")          # a name we never armed (a late echo after clear) changes nothing
    assert tr.heard() == ("Hello there friend.", True)
    tr.note_mark("m2")
    text, cut = tr.heard()
    assert text == "Hello there friend. How can I help you today?" and not cut
    assert not tr.is_playing() and tr.buffered_ms() == 0


def test_reset_forgets_marks_so_a_late_echo_is_not_what_was_heard():
    clk = Clock()
    tr = PlaybackTracker(clk)
    tr.on_sent("Hello there friend.", 1000)
    tr.arm_mark("m1")
    tr.reset()
    tr.note_mark("m1")
    tr.on_sent("Next sentence here.", 1000)
    clk.t = 500
    assert tr.heard() == ("Next", True)     # still the clock; the stale mark did not enter mark mode


def _mark_sending(r: Rig) -> None:
    """Send callback that names each chunk the way Twilio's m{n} marks do."""
    async def send(pcm, rate):
        r.sent.append((r.clk.t, len(pcm), rate))
        return f"m{len(r.sent)}"

    r.s.send_audio = send


def _hangup_with_marks():
    ended = []

    async def hang(args):
        async def after():
            ended.append(1)
        return ToolResult({"ok": True}, after=after, strict=True)

    llm = FakeLLM([["Goodbye ", "now. ", ToolCall("h", "hang", {})], ["Bye ", "then."]])
    r = Rig(llm, tts=FakeTTS(word_ms=400), tools={"hang": hang},
            cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500))
    _mark_sending(r)
    return r, ended


async def test_hangup_waits_for_the_playback_mark_after_the_clock_has_passed():
    r, ended = _hangup_with_marks()
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.tts.texts == ["Goodbye now.", "Bye then."]
    assert set(r.s.tracker.pending_marks) == {"m1", "m2"} and ended == []
    # Clock is already past the estimated end, still inside the mark timeout: the line stays up.
    r.clk.t = r.s.tracker.ends_at() + 100
    await asyncio.sleep(0.25)
    # Clock buffer is already empty. The hang-up still waits because the marks are unacked.
    assert ended == [] and r.s.tracker.buffered_ms() == 0 and r.s.tracker.is_playing()
    for name in list(r.s.tracker.pending_marks):
        r.s.tracker.note_mark(name)
    await asyncio.sleep(0.25)
    assert ended == [1]


async def test_hangup_runs_when_a_playback_mark_never_arrives():
    r, ended = _hangup_with_marks()
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert ended == [] and r.s.tracker.pending_marks
    r.clk.t = r.s.tracker.ends_at() + r.s.cfg.mark_timeout_ms + 1
    await asyncio.sleep(0.25)
    assert ended == [1]


async def test_say_and_wait_holds_for_marks_after_the_clock_is_past():
    """Voicemail plays with say_and_wait and then hangs up. Before the first ack, buffered_ms
    follows the local clock, so that hang-up must not run while the mark is still pending."""
    r = Rig(FakeLLM([["x."]]), tts=FakeTTS(word_ms=100),
            cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500))
    _mark_sending(r)
    task = asyncio.create_task(r.s.say_and_wait("Please call us back."))
    await r.settle()
    assert r.s.tracker.pending_marks
    r.clk.t = r.s.tracker.ends_at() + 100
    await asyncio.sleep(0.25)
    assert r.s.tracker.buffered_ms() == 0 and not task.done()
    for name in list(r.s.tracker.pending_marks):
        r.s.tracker.note_mark(name)
    await asyncio.sleep(0.25)
    assert task.done() and not task.cancelled()
    await task


async def test_say_and_wait_returns_when_the_mark_never_arrives():
    r = Rig(FakeLLM([["x."]]), tts=FakeTTS(word_ms=100),
            cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500))
    _mark_sending(r)
    task = asyncio.create_task(r.s.say_and_wait("Please call us back."))
    await r.settle()
    assert r.s.tracker.pending_marks
    r.clk.t = r.s.tracker.ends_at() + r.s.cfg.mark_timeout_ms + 1
    await asyncio.sleep(0.25)
    assert task.done() and not task.cancelled()
    await task


def test_twilio_session_waits_on_marks_and_a_browser_session_does_not():
    store = Store()
    rt = Runtime(ApiLLM(), lambda v: ApiTTS(), FakeVAD, store)
    tid = store.create_tenant("t")

    class Twilioish:
        out_rate = 8000
        call_sid = "CA1"

        def __init__(self):
            self.handler = None

        async def send_audio(self, pcm, rate):
            return None

        async def clear(self):
            return None

        async def end(self):
            return None

        def set_mark_handler(self, fn):
            self.handler = fn

    class Browserish:
        out_rate = 24000

        async def send_audio(self, pcm, rate):
            return None

        async def clear(self):
            return None

        async def end(self):
            return None

    phone = Twilioish()
    call = store.create_call(tid, None, "twilio", {"systemPrompt": "p"}, {})
    sess = rt.build_session(call, phone)
    assert sess.cfg.output_latency_ms == 200
    assert phone.handler == sess.tracker.note_mark
    web = store.create_call(tid, None, "web", {"systemPrompt": "p"}, {})
    assert rt.build_session(web, Browserish()).cfg.output_latency_ms == 0
    # 8 kHz without a mark handler (AudioSocket) stays on the clock, same as the unit-test rigs.
    class PhoneNoMarks:
        out_rate = 8000

        async def send_audio(self, pcm, rate):
            return None

        async def clear(self):
            return None

        async def end(self):
            return None
    other = store.create_call(tid, None, "sip", {"systemPrompt": "p"}, {})
    assert rt.build_session(other, PhoneNoMarks()).cfg.output_latency_ms == 0


def test_missing_mark_stops_blocking_after_the_timeout_but_a_late_echo_still_counts():
    clk = Clock()
    tr = PlaybackTracker(clk, mark_timeout_ms=1500)
    tr.on_sent("Hello there friend.", 1000)
    tr.arm_mark("m1")
    tr.on_sent("How can I help you today?", 2000)
    tr.arm_mark("m2")
    tr.note_mark("m1")
    clk.t = tr.ends_at() + 100
    assert tr.is_playing() and tr.buffered_ms() == 2000
    clk.t = tr.ends_at() + 1500
    assert not tr.is_playing() and tr.buffered_ms() == 0
    assert tr.heard() == ("Hello there friend.", True)
    tr.note_mark("m2")  # armed through the give-up; the echo is late, not unknown
    assert tr.heard() == ("Hello there friend. How can I help you today?", False)
    assert not tr.is_playing() and tr.buffered_ms() == 0


def _clock_heard(chunks, now):
    parts: list[str] = []
    cut = False
    for c in chunks:
        if now >= c.start_ms + c.dur_ms:
            parts.append(c.text)
        elif now > c.start_ms:
            words = c.text.split()
            k = int(len(words) * ((now - c.start_ms) / c.dur_ms))
            if k > 0:
                parts.append(" ".join(words[:k]))
            cut = True
            break
        else:
            cut = True
            break
    return " ".join(parts).strip(), cut


async def test_acked_mark_does_not_block_the_tool_followup_past_the_timeout():
    """Once a mark has been acked, unacked audio still counts against lookahead — but only until
    mark_timeout_ms after the clock estimate. The deferred hang-up then runs with no further ack."""
    ended = []

    async def hang(args):
        async def after():
            ended.append(1)
        return ToolResult({"ok": True}, after=after, strict=True)

    llm = FakeLLM([
        ["Goodbye now. ", "Please hold the line a moment longer now. ", ToolCall("h", "hang", {})],
        ["Thanks for calling today friend."],
    ])
    r = Rig(llm, tts=FakeTTS(word_ms=200), tools={"hang": hang},
            cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500, tts_lookahead_ms=1500))
    _mark_sending(r)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.tts.texts == ["Goodbye now.", "Please hold the line a moment longer now."]
    assert set(r.s.tracker.pending_marks) == {"m1", "m2"} and not r.s._gen.produce_done
    r.s.tracker.note_mark("m1")
    # Clock buffer would already release the follow-up; the unacked mark must still hold it.
    r.clk.t = r.s.tracker.ends_at() - 100
    assert r.s.tracker.ends_at() - r.clk.t < r.s.cfg.tts_lookahead_ms
    assert r.s.tracker.buffered_ms() > r.s.cfg.tts_lookahead_ms and r.s.tracker.is_playing()
    await asyncio.sleep(0.2)
    assert r.tts.texts == ["Goodbye now.", "Please hold the line a moment longer now."]
    assert ended == [] and not r.s._gen.produce_done
    r.clk.t = r.s.tracker.ends_at() + r.s.cfg.mark_timeout_ms
    await asyncio.sleep(0.2)
    assert r.tts.texts == ["Goodbye now.", "Please hold the line a moment longer now.",
                           "Thanks for calling today friend."]
    assert ended == [] and r.s.tracker.pending_marks
    r.clk.t = r.s.tracker.ends_at() + r.s.cfg.mark_timeout_ms
    await asyncio.sleep(0.2)
    assert ended == [1] and r.s.tracker.pending_marks


async def test_lost_last_mark_does_not_barge_in_after_the_timeout():
    llm = FakeLLM([
        ["Hello there friend. ", "How can I help you today?"],
        ["Next reply here."],
    ])
    r = Rig(llm, cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500))
    _mark_sending(r)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.tts.texts == ["Hello there friend.", "How can I help you today?"]
    r.s.tracker.note_mark("m1")
    assert r.s.tracker.is_playing() and r.s.tracker.buffered_ms() == 480
    r.clk.t = r.s.tracker.ends_at() + r.s.cfg.mark_timeout_ms
    assert not r.s.tracker.is_playing() and r.s.tracker.buffered_ms() == 0
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.clears == 0 and r.s.stats["interruptions"] == 0
    assistants = [m["content"] for m in r.s.history if m["role"] == "assistant"]
    assert assistants == ["Hello there friend. How can I help you today?", "Next reply here."]


async def test_later_turn_barge_in_hears_only_that_turn_until_its_own_mark():
    llm = FakeLLM([
        ["Hello there friend."],
        ["One two three four five. ", "Six seven eight nine ten."],
    ])
    r = Rig(llm, tts=FakeTTS(word_ms=200), cfg=SessionConfig(tts_poll_s=0, mark_timeout_ms=1500))
    _mark_sending(r)
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    assert r.tts.texts == ["Hello there friend."]
    for name in list(r.s.tracker.pending_marks):
        r.s.tracker.note_mark(name)
    assert not r.s.tracker.is_playing() and not r.s._agent_active()
    await r.frames("s", 15)
    await r.frames("q", 22)
    await r.settle()
    chunks = list(r.s.tracker._chunks)
    assert [c.text for c in chunks] == ["One two three four five.", "Six seven eight nine ten."]
    # No ack for this playback yet: the buffer follows the clock, not the previous turn's marks.
    assert r.s.tracker.buffered_ms() == r.s.tracker.ends_at() - r.clk.t
    assert r.s.tracker.heard()[0] != "Hello there friend."
    second = chunks[1]
    n_quiet = round((second.start_ms + 400 - r.clk.t - 5 * 32) / 32)
    assert n_quiet > 0
    interrupt_at = r.clk.t + (n_quiet + 5) * 32
    expect, expect_cut = _clock_heard(chunks, interrupt_at)
    assert expect.startswith("One two three four five.") and "Hello" not in expect and expect_cut
    await r.frames("q", n_quiet)
    await r.frames("s", 6)
    await r.settle()
    ev = r.last("user_interrupted")
    assert ev["t"] == interrupt_at
    assert ev["heard"] == expect and ev["cut"] is True
    assert r.s.history[-3]["content"] == "Hello there friend."
    assert r.s.history[-1]["content"] == expect + " [interrupted]"


async def test_say_and_inject_drop_a_finished_playback_before_their_own_audio():
    r = Rig(FakeLLM([["One two three four five."]]), tts=FakeTTS(word_ms=200),
            cfg=SessionConfig(tts_poll_s=0))
    _mark_sending(r)
    r.s.say("Hello there friend.")
    await r.settle()
    for name in list(r.s.tracker.pending_marks):
        r.s.tracker.note_mark(name)
    assert not r.s.tracker.is_playing()
    r.clk.t = 5000
    r.s.say("One two three four five.")
    await r.settle()
    assert [c.text for c in r.s.tracker._chunks] == ["One two three four five."]
    assert "Hello" not in r.s.tracker.heard()[0]

    r2 = Rig(FakeLLM([["One two three four five."]]), tts=FakeTTS(word_ms=200),
             cfg=SessionConfig(tts_poll_s=0))
    _mark_sending(r2)
    r2.s.tracker.on_sent("Hello there friend.", 600)
    r2.s.tracker.arm_mark("old")
    r2.s.tracker.note_mark("old")
    r2.clk.t = 5000
    await r2.s.inject_text("42")
    await r2.settle()
    assert [c.text for c in r2.s.tracker._chunks] == ["One two three four five."]
    assert "Hello" not in r2.s.tracker.heard()[0]
