from core.turn import Ev, TurnConfig, TurnManager

F = 32.0  # ms per frame


def feed(tm, pattern):
    """pattern: list of (is_speech, n_frames). Returns [(kind, t_ms), ...]."""
    out = []
    for speech, n in pattern:
        for _ in range(n):
            for e in tm.process(0.9 if speech else 0.1):
                out.append((e.kind, e.t_ms))
    return out


def kinds(events):
    return [k for k, _ in events]


def test_simple_turn_timing():
    # 30 speech frames, then silence. Speculate after 192 ms, end after 640 ms of silence.
    ev = feed(TurnManager(), [(True, 30), (False, 30)])
    assert kinds(ev) == [Ev.SPEECH_START, Ev.SPECULATE, Ev.END_OF_TURN]
    t = dict((k, t) for k, t in ev)
    assert t[Ev.SPEECH_START] == 2 * F                     # 3rd speech frame reaches 96 ms
    assert t[Ev.SPECULATE] == (29 + 6) * F                 # 6th silence frame = 192 ms
    assert t[Ev.END_OF_TURN] == (29 + 20) * F              # 20th silence frame = 640 ms


def test_pause_then_resume_cancels_speculation():
    ev = feed(TurnManager(), [(True, 20), (False, 8), (True, 10), (False, 30)])
    assert kinds(ev) == [Ev.SPEECH_START, Ev.SPECULATE, Ev.SPEECH_RESUME,
                         Ev.SPECULATE, Ev.END_OF_TURN]


def test_confident_scorer_ends_turn_earlier():
    calls = []
    tm = TurnManager(eot_scorer=lambda: (calls.append(1), 0.9)[1])
    ev = feed(tm, [(True, 20), (False, 30)])
    t = dict((k, t) for k, t in ev)
    assert t[Ev.END_OF_TURN] == (19 + 10) * F              # 320 ms of silence
    assert len(calls) == 1                                 # consulted once per pause


def test_unconfident_scorer_waits_full_end_ms():
    tm = TurnManager(eot_scorer=lambda: 0.2)
    t = dict((k, t) for k, t in feed(tm, [(True, 20), (False, 30)]))
    assert t[Ev.END_OF_TURN] == (19 + 20) * F


def test_short_noise_burst_is_ignored():
    assert feed(TurnManager(), [(False, 5), (True, 2), (False, 40)]) == []


def test_burst_just_over_start_threshold_is_aborted_as_noise():
    tm = TurnManager()
    ev = feed(tm, [(True, 4), (False, 10)])  # 128 ms < min_speech_ms (160)
    assert kinds(ev) == [Ev.SPEECH_START, Ev.SPEECH_ABORT]
    assert not tm.in_turn


def test_barge_in_needs_longer_speech_while_agent_speaks():
    # While the agent talks, SPEECH_START waits for backchannel_ms (450), not the old 192 ms
    # barge floor, so a short "yeah" never opens a turn (no start, and therefore no abort).
    tm = TurnManager()
    tm.set_agent_speaking(True)
    # 448 ms ends before the 450 ms gate: not a turn yet, but the session must hear the words.
    ev = feed(tm, [(True, 14), (False, 10)])
    # The check waits out speculate_ms (6 frames), not the first silent frame.
    assert kinds(ev) == [Ev.OVERLAP_CHECK] and ev[0][1] == (14 + 5) * F
    tm2 = TurnManager()
    tm2.set_agent_speaking(True)
    ev = feed(tm2, [(True, 16)])                            # 512 ms: real interruption
    assert kinds(ev) == [Ev.SPEECH_START]
    assert ev[0][1] == 14 * F                               # 15th frame: 480 ms of speech >= 450


def test_overlap_check_reports_how_long_the_burst_was():
    tm = TurnManager()
    tm.set_agent_speaking(True)
    for _ in range(14):
        assert tm.process(0.9) == []
    for _ in range(5):
        assert tm.process(0.1) == []
    ev = tm.process(0.1)
    assert len(ev) == 1 and ev[0].kind is Ev.OVERLAP_CHECK and ev[0].speech_ms == 14 * F
    high = TurnManager(TurnConfig(backchannel_ms=192))
    high.set_agent_speaking(True)
    opened = []
    for _ in range(8):
        opened += high.process(0.9)
    opened += high.process(0.1)
    assert Ev.OVERLAP_CHECK not in [e.kind for e in opened]
    assert opened[0].kind is Ev.SPEECH_START


def test_short_barge_aborts_and_a_long_one_does_not():
    # The default 450 ms gate never opens a turn under barge_confirm_ms (400), so a cough
    # there is an overlap check, not an abort. bargeIn=high lowers the gate to 192 ms:
    # 224 ms starts and then aborts, 448 ms is past 400 and speculates.
    cfg = TurnConfig(backchannel_ms=192)
    tm = TurnManager(cfg)
    tm.set_agent_speaking(True)
    ev = feed(tm, [(True, 7), (False, 6)])
    assert kinds(ev) == [Ev.SPEECH_START, Ev.SPEECH_ABORT]
    assert not tm.in_turn
    tm2 = TurnManager(cfg)
    tm2.set_agent_speaking(True)
    ev2 = feed(tm2, [(True, 14), (False, 6)])
    assert kinds(ev2) == [Ev.SPEECH_START, Ev.SPECULATE]


def test_onset_time_is_reported():
    tm = TurnManager()
    events = []
    for p in [0.1] * 5 + [0.9] * 10 + [0.1] * 30:
        events += tm.process(p)
    start = next(e for e in events if e.kind is Ev.SPEECH_START)
    assert start.onset_ms == 5 * F
    end = next(e for e in events if e.kind is Ev.END_OF_TURN)
    assert end.onset_ms == 5 * F


def test_manager_is_reusable_across_turns():
    ev = feed(TurnManager(), [(True, 15), (False, 30), (True, 15), (False, 30)])
    assert kinds(ev) == [Ev.SPEECH_START, Ev.SPECULATE, Ev.END_OF_TURN] * 2


def test_late_confident_verdict_shortens_the_wait():
    tm = TurnManager()
    ev = feed(tm, [(True, 20), (False, 8)])                # speculating, 256 ms of silence
    assert kinds(ev) == [Ev.SPEECH_START, Ev.SPECULATE]
    assert tm.apply_eot(0.9) == []                          # target now 320 ms, not reached yet
    ev = feed(tm, [(False, 5)])
    assert kinds(ev) == [Ev.END_OF_TURN]
    assert ev[0][1] == (19 + 10) * F                        # ended at 320 ms of silence


def test_verdict_arriving_after_the_new_target_ends_immediately():
    tm = TurnManager()
    feed(tm, [(True, 20), (False, 14)])                     # 448 ms of silence, not yet 640
    ev = tm.apply_eot(0.95)
    assert [e.kind for e in ev] == [Ev.END_OF_TURN] and not tm.in_turn


def test_confident_not_done_extends_the_wait_for_slow_speakers():
    tm = TurnManager()
    feed(tm, [(True, 20), (False, 8)])
    assert tm.apply_eot(0.05) == []
    ev = feed(tm, [(False, 30)])                            # 38 frames of silence = 1216 ms in total
    t = dict((k, t) for k, t in ev)
    assert t[Ev.END_OF_TURN] == (19 + 32) * F               # 32nd silence frame = 1024 ms >= 1000


def test_middling_verdict_changes_nothing_and_verdict_outside_a_pause_is_ignored():
    tm = TurnManager()
    assert tm.apply_eot(0.99) == []                         # not in a turn
    feed(tm, [(True, 20)])
    assert tm.apply_eot(0.99) == []                         # speaking, not paused
    feed(tm, [(False, 8)])
    assert tm.apply_eot(0.5) == []
    ev = feed(tm, [(False, 20)])
    assert dict((k, t) for k, t in ev)[Ev.END_OF_TURN] == (19 + 20) * F


def test_verdict_after_speech_resumes_is_ignored():
    tm = TurnManager()
    feed(tm, [(True, 20), (False, 8), (True, 5)])           # caller resumed: speculation cancelled
    assert tm.apply_eot(0.99) == []


def test_one_silent_frame_does_not_split_a_held_barge():
    # 8 speech frames (256 ms) are already past the 192 ms floor. One silent frame is a VAD
    # dip, not the end of the burst: the following speech is the same run.
    tm = TurnManager()
    tm.set_agent_speaking(True)
    ev = feed(tm, [(True, 8), (False, 1), (True, 7)])       # 15 speech frames = 480 ms
    assert kinds(ev) == [Ev.SPEECH_START] and ev[0][1] == 15 * F

    tm = TurnManager()
    tm.set_agent_speaking(True)
    events = []
    for p in [0.9] * 8 + [0.1] + [0.9] * 5 + [0.1] * 5:
        events += tm.process(p)
    assert events == []                                     # pause has not reached speculate_ms
    events = tm.process(0.1)                                # 6th silent frame = 192 ms
    assert len(events) == 1 and events[0].kind is Ev.OVERLAP_CHECK
    assert events[0].speech_ms == 13 * F                    # 8 + 5 frames, the dip is not speech


def test_subfloor_cough_is_not_glued_onto_the_next_burst():
    tm = TurnManager()
    tm.set_agent_speaking(True)
    events = []
    for p in [0.9] * 4 + [0.1] * 6 + [0.9] * 12 + [0.1] * 6:  # 128 ms cough, then 384 ms
        events += tm.process(p)
    assert [e.kind for e in events] == [Ev.OVERLAP_CHECK]
    assert events[0].speech_ms == 12 * F


def test_playback_ending_promotes_a_held_run_instead_of_dropping_it():
    tm = TurnManager()
    tm.set_agent_speaking(True)
    assert feed(tm, [(True, 12)]) == []                     # 384 ms, still under the 450 ms gate
    tm.set_agent_speaking(False)
    ev = tm.process(0.1)
    assert [e.kind for e in ev] == [Ev.SPEECH_START] and tm.in_turn
    ev = feed(tm, [(False, 25)])
    assert kinds(ev) == [Ev.SPECULATE, Ev.END_OF_TURN]
