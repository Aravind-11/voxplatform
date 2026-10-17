from core.chunker import SentenceChunker


def run(deltas, **kw):
    c = SentenceChunker(**kw)
    out = []
    for d in deltas:
        out += c.push(d)
    out += c.flush()
    return out


def split_every(s, n):
    return [s[i : i + n] for i in range(0, len(s), n)]


def test_basic_sentences_any_delta_size():
    text = "Hello there. How are you? I'm fine!"
    for n in (1, 2, 3, 7, 100):
        assert run(split_every(text, n)) == ["Hello there.", "How are you?", "I'm fine!"]


def test_first_chunk_splits_early_at_clause():
    text = "I understand your concern about this, and here is what we can do. Next."
    out = run(split_every(text, 4))
    assert out[0] == "I understand your concern about this,"
    assert out[1] == "and here is what we can do."
    assert out[2] == "Next."


def test_first_chunk_cuts_at_twelve_words_without_punctuation():
    text = "Thanks for calling I will check that for you right away."
    out = run(split_every(text, 3))
    assert len(out[0].split()) <= 12
    assert " ".join(out) == text
    # Eleven words stays one chunk. A longer unpunctuated reply cuts on the word boundary.
    longer = text[:-1] + " please hold for just a moment"
    c = SentenceChunker()
    early = []
    for d in split_every(longer, 4):
        early += c.push(d)
    rest = c.flush()
    assert len(early) == 1 and len(early[0].split()) == 12
    assert rest == [" ".join(longer.split()[12:])]
    assert early[0] + " " + rest[0] == longer
    partial = " ".join(longer.split()[:11]) + " " + longer.split()[11][:3]
    assert SentenceChunker().push(partial) == []


def test_word_cap_does_not_apply_after_the_first_chunk():
    second = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    assert run(["Hello there. " + second]) == ["Hello there.", second]


def test_short_clause_is_not_split_early():
    out = run(["Well, I think so. Okay."])
    assert out == ["Well, I think so.", "Okay."]


def test_later_chunks_do_not_split_on_commas():
    out = run(["First sentence here is long enough. Then a, b, c, and d together. End."])
    assert out == ["First sentence here is long enough.", "Then a, b, c, and d together.", "End."]


def test_abbreviations_and_decimals_do_not_split():
    out = run(split_every("Dr. Smith paid 3.5 dollars. Next one.", 3))
    assert out == ["Dr. Smith paid 3.5 dollars.", "Next one."]


def test_markdown_is_not_spoken():
    out = run(["**Bold** text and `code`.\n- item one\n- item two"])
    joined = " ".join(out)
    for bad in "*`#_>":
        assert bad not in joined
    assert out == ["Bold text and code.", "item one", "item two"]


def test_think_blocks_are_dropped_even_when_split_across_deltas():
    text = "<think>secret reasoning. more.</think>Hello world."
    for n in (1, 3, 5, 100):
        assert run(split_every(text, n)) == ["Hello world."]


def test_unterminated_think_block_is_never_emitted():
    assert run(["Visible one. <think>still thinking"]) == ["Visible one."]


def test_no_empty_chunks():
    out = run(["", "  ", "\n", "Hi.", " ", "\n\n"])
    assert out == ["Hi."]
