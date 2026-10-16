"""Turn-taking state machine driven by per-frame speech probabilities.

Pure logic, no I/O, time = frame_index * frame_ms, so it is deterministic and testable.

Events
  SPEECH_START   caller started talking (enough consecutive speech frames)
  SPECULATE      short silence after speech: start generating a reply speculatively
  SPEECH_RESUME  caller resumed before the turn ended: cancel the speculative reply
  END_OF_TURN    silence long enough: commit the reply and let it play
  SPEECH_ABORT   a started "turn" was too short to be real speech (noise): discard

An optional end-of-turn scorer (e.g. UltraVAD) is consulted once per pause. If it is
confident the caller is done, the turn ends after `fast_end_ms` of silence instead of
`end_ms`, trading nothing in accuracy for lower latency on clear endings.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class Ev(str, Enum):
    SPEECH_START = "speech_start"
    SPECULATE = "speculate"
    SPEECH_RESUME = "speech_resume"
    END_OF_TURN = "end_of_turn"
    SPEECH_ABORT = "speech_abort"


@dataclass(frozen=True)
class TurnEvent:
    kind: Ev
    t_ms: float            # time the event was emitted
    onset_ms: float = 0.0  # time the current utterance's speech actually began


@dataclass
class TurnConfig:
    frame_ms: float = 32.0              # Silero VAD frame at 16 kHz = 512 samples
    threshold: float = 0.5
    start_ms: float = 96.0              # consecutive speech needed to start a turn
    start_ms_agent_speaking: float = 192.0  # stricter while the agent talks (barge-in)
    min_speech_ms: float = 160.0        # shorter utterances are treated as noise
    speculate_ms: float = 192.0         # silence before speculative generation starts
    fast_end_ms: float = 320.0          # end of turn when the scorer is confident
    end_ms: float = 640.0               # end of turn otherwise
    eot_threshold: float = 0.7
    # Verdict-driven timing (apply_eot): a confident "done" ends the turn after fast_end_ms; a confident
    # "not done" (trailing off, mid-sentence) waits patient_end_ms so slow speakers are not cut off.
    patient_threshold: float = 0.25
    patient_end_ms: float = 1000.0


class TurnManager:
    def __init__(self, cfg: TurnConfig | None = None,
                 eot_scorer: Optional[Callable[[], float]] = None):
        self.cfg = cfg or TurnConfig()
        self.eot_scorer = eot_scorer
        self.agent_speaking = False
        self._frame = 0
        self._reset()

    def _reset(self) -> None:
        self._speaking = False
        self._run_ms = 0.0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speculated = False
        self._end_target = self.cfg.end_ms
        self._onset_ms = 0.0
        self._run_start_ms = 0.0

    def apply_eot(self, score: float) -> list[TurnEvent]:
        """Feed a late end-of-turn verdict (P(caller is done), 0..1) from an async scorer.

        Only meaningful during a pause (after SPECULATE). Never makes the turn end later than the
        default except for a confident "not done"; if the new target is already behind us the turn
        ends immediately.
        """
        c = self.cfg
        if not (self._speaking and self._speculated):
            return []
        if score >= c.eot_threshold:
            self._end_target = min(self._end_target, c.fast_end_ms)
        elif score <= c.patient_threshold:
            self._end_target = max(self._end_target, c.patient_end_ms)
        else:
            return []
        if self._silence_ms >= self._end_target:
            t = self._frame * c.frame_ms
            ev = [TurnEvent(Ev.END_OF_TURN, t, self._onset_ms)]
            self._reset()
            return ev
        return []

    def set_agent_speaking(self, speaking: bool) -> None:
        self.agent_speaking = speaking

    @property
    def in_turn(self) -> bool:
        return self._speaking

    def process(self, speech_prob: float) -> list[TurnEvent]:
        c = self.cfg
        t = self._frame * c.frame_ms
        self._frame += 1
        is_speech = speech_prob >= c.threshold
        out: list[TurnEvent] = []

        if not self._speaking:
            if is_speech:
                if self._run_ms == 0.0:
                    self._run_start_ms = t
                self._run_ms += c.frame_ms
                need = c.start_ms_agent_speaking if self.agent_speaking else c.start_ms
                if self._run_ms >= need:
                    self._speaking = True
                    self._onset_ms = self._run_start_ms
                    self._speech_ms = self._run_ms
                    self._silence_ms = 0.0
                    out.append(TurnEvent(Ev.SPEECH_START, t, self._onset_ms))
            else:
                self._run_ms = 0.0
            return out

        # speaking
        if is_speech:
            self._speech_ms += c.frame_ms
            self._silence_ms = 0.0
            if self._speculated:
                out.append(TurnEvent(Ev.SPEECH_RESUME, t, self._onset_ms))
                self._speculated = False
                self._end_target = c.end_ms
            return out

        self._silence_ms += c.frame_ms
        if self._speech_ms < c.min_speech_ms and self._silence_ms >= c.speculate_ms:
            out.append(TurnEvent(Ev.SPEECH_ABORT, t, self._onset_ms))
            self._reset()
            return out
        if not self._speculated and self._silence_ms >= c.speculate_ms:
            self._speculated = True
            confident = bool(self.eot_scorer and self.eot_scorer() >= c.eot_threshold)
            self._end_target = c.fast_end_ms if confident else c.end_ms
            out.append(TurnEvent(Ev.SPECULATE, t, self._onset_ms))
        if self._speculated and self._silence_ms >= self._end_target:
            out.append(TurnEvent(Ev.END_OF_TURN, t, self._onset_ms))
            self._reset()
        return out
