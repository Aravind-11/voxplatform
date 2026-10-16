import base64
import io
import wave
from types import SimpleNamespace as NS

import numpy as np

from core.llm import TRANSCRIBE_PROMPT, VllmClient
from core.session import ToolCall


def chunk(content=None, tool_calls=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])


def tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class FakeCompletions:
    def __init__(self, chunks=None, reply=""):
        self.chunks, self.reply, self.kwargs, self.calls = chunks or [], reply, None, 0

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls += 1
        if kwargs.get("stream"):
            async def gen():
                for c in self.chunks:
                    yield c
            return gen()
        reply = self.reply.pop(0) if isinstance(self.reply, list) else self.reply
        return NS(choices=[NS(message=NS(content=reply))])


def make(chunks=None, reply="", style="input_audio"):
    comp = FakeCompletions(chunks, reply)
    client = NS(chat=NS(completions=comp))
    return VllmClient(client=client, audio_style=style), comp


AUDIO = (np.arange(1600) % 200).astype(np.int16)


def test_audio_part_is_valid_16k_mono_wav():
    llm, _ = make()
    part = llm.audio_part(AUDIO)
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"]))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 1600)
        assert np.array_equal(np.frombuffer(w.readframes(1600), dtype="<i2"), AUDIO)


def test_audio_url_fallback_uses_data_uri():
    llm, _ = make(style="audio_url")
    part = llm.audio_part(AUDIO)
    assert part["type"] == "audio_url" and part["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_message_order_history_then_audio_then_followups():
    llm, _ = make()
    hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    fups = [{"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "1", "content": "{}"}]
    msgs = llm.build_messages(hist, AUDIO, fups)
    assert msgs[:3] == hist and msgs[3]["role"] == "user" and msgs[3]["content"][0]["type"] == "input_audio"
    assert msgs[4:] == fups


async def test_stream_yields_text_and_passes_options():
    llm, comp = make([chunk("Hel"), chunk("lo"), NS(choices=[])])
    out = [x async for x in llm.stream([], AUDIO, [], [])]
    assert out == ["Hel", "lo"]
    assert comp.kwargs["stream"] is True and comp.kwargs["model"] == "ultravox"
    assert "tools" not in comp.kwargs
    llm2, comp2 = make([chunk("x")])
    _ = [x async for x in llm2.stream([], AUDIO, [], [{"type": "function"}])]
    assert comp2.kwargs["tools"] == [{"type": "function"}]


async def test_stream_assembles_fragmented_tool_calls():
    llm, _ = make([chunk("Checking. "),
                   chunk(tool_calls=[tc(0, id="c1", name="lo", args='{"da')]),
                   chunk(tool_calls=[tc(0, name="okup", args='y": "mon"}')]),
                   chunk(tool_calls=[tc(1, id="c2", name="ping", args="")])])
    out = [x async for x in llm.stream([], AUDIO, [], [{"type": "function"}])]
    assert out[0] == "Checking. "
    assert out[1] == ToolCall("c1", "lookup", {"day": "mon"})
    assert out[2] == ToolCall("c2", "ping", {})


async def test_malformed_tool_arguments_become_empty_dict():
    llm, _ = make([chunk(tool_calls=[tc(0, id="c1", name="f", args='{"broken')])])
    out = [x async for x in llm.stream([], AUDIO, [], [{"type": "function"}])]
    assert out == [ToolCall("c1", "f", {})]


async def test_transcribe_sends_prompt_and_audio():
    llm, comp = make(reply="hello there")
    assert await llm.transcribe(AUDIO) == "hello there"
    assert comp.kwargs["messages"][0]["role"] == "system"
    content = comp.kwargs["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": TRANSCRIBE_PROMPT}
    assert content[1]["type"] == "input_audio" and comp.kwargs["temperature"] == 0


async def test_thinking_disabled_and_stripped_from_transcripts():
    llm, comp = make(reply="<think>\nhmm\n</think>hello")
    assert await llm.transcribe(AUDIO) == "hello"
    assert comp.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    llm2, comp2 = make(reply="<think>never closed")
    assert await llm2.transcribe(AUDIO) == ""
    _ = [x async for x in llm2.stream([], AUDIO, [], [])]
    assert comp2.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def lp(**probs):
    import math
    return NS(content=[NS(top_logprobs=[NS(token=t, logprob=math.log(p)) for t, p in probs.items()])])


async def test_eot_score_reads_yes_no_confidence_from_logprobs():
    llm, comp = make()
    comp.reply = None
    async def create(**kw):
        comp.kwargs = kw
        return NS(choices=[NS(message=NS(content="yes"), logprobs=lp(yes=0.6, Yes=0.2, no=0.15, other=0.05))])
    comp.create = create
    hist = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "Which day works?"}]
    score = await llm.eot_score(hist, AUDIO)
    assert abs(score - 0.8 / 0.95) < 1e-6
    kw = comp.kwargs
    assert kw["max_tokens"] == 1 and kw["logprobs"] is True and kw["temperature"] == 0
    text = kw["messages"][0]["content"][0]["text"]
    assert "Which day works?" in text and kw["messages"][0]["content"][1]["type"] == "input_audio"


async def test_eot_score_falls_back_to_the_text_answer_without_logprobs():
    llm, comp = make()
    for word, want in (("Yes.", 1.0), ("no", 0.0), ("maybe", 0.5)):
        async def create(word=word, **kw):
            return NS(choices=[NS(message=NS(content=word), logprobs=None)])
        comp.create = create
        assert await llm.eot_score([], AUDIO) == want


async def test_transcribe_rejects_answers_that_are_longer_than_the_audio():
    answer = "I am not located anywhere. I am an AI language model and do not have a physical location."
    llm, comp = make(reply=answer)                 # 0.1 s of audio can't hold 17 words
    assert await llm.transcribe(AUDIO) == ""       # dropped; the session records [inaudible]
    assert comp.calls == 2                         # one stricter retry first


async def test_transcribe_retry_can_recover():
    llm, comp = make(reply=["I am an AI language model and I do not have a location at all.", "where are you"])
    assert await llm.transcribe(np.zeros(8000, np.int16)) == "where are you"   # 0.5 s: at most 4 words
    assert comp.calls == 2


async def test_eot_prompt_version_is_selectable():
    from core.llm import EOT_PROMPT, EOT_PROMPT_V2
    llm, comp = make(); comp.reply = "yes"
    await llm.eot_score([], AUDIO)
    assert comp.kwargs["messages"][0]["content"][0]["text"] == EOT_PROMPT.format(ctx="")
    client = NS(chat=NS(completions=comp))
    llm2 = VllmClient(client=client, eot_prompt="v2")
    await llm2.eot_score([], AUDIO)
    assert comp.kwargs["messages"][0]["content"][0]["text"] == EOT_PROMPT_V2.format(ctx="")
