from core.barge import PlaybackTracker, trim_history


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
