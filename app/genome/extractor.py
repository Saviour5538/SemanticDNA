from app.genome import trigram as tg
from app.genome import phonetic as ph
from app.genome import context_depth as cd
from app.genome import entropy as en
from app.genome.pos_tagger import compute_all_pos_tags, get_nlp

_PAD = "__PAD__"

# ------------------------------------------------------------------ #
# Internal helpers                                                      #
# ------------------------------------------------------------------ #

def _filter_spacy_doc(spacy_doc) -> list[tuple[str, str]]:
    """Extract (token, pos) pairs from a spaCy Doc, dropping spaces/punctuation."""
    return [
        (t.text, t.pos_)
        for t in spacy_doc
        if not t.is_space and t.text.strip() and (t.text.isalpha() or "_" in t.text)
    ]


def _build_genome_from_parsed(
    filtered: list[tuple[str, str]],
    corpus_stats: dict[str, int],
    total_docs: int,
    smoothing: float,
) -> list[dict]:
    """Build a gene array from pre-parsed (token, pos_tag) pairs and pre-fetched IDF stats."""
    if not filtered:
        return []

    tokens = [t for t, _ in filtered]
    pos_tags = [p for _, p in filtered]

    if len(tokens) == 1:
        tokens = [_PAD] + tokens + [_PAD]
        pos_tags = ["X"] + pos_tags + ["X"]
    elif len(tokens) == 2:
        tokens = tokens + [_PAD]
        pos_tags = pos_tags + ["X"]

    genome: list[dict] = []
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
        p1, p2, p3 = pos_tags[i], pos_tags[i + 1], pos_tags[i + 2]
        genome.append({
            "trigram_hash":  tg.hash_trigram(w1, w2, w3),
            "phonetic_code": ph.compute_phonetic_sequence([w1, w2, w3]),
            "pos_sequence":  f"{p1}-{p2}-{p3}",
            "context_depth": cd.compute_depth_for_window([w1, w2, w3]),
            "entropy_score": en.compute_entropy_for_window(
                [w1, w2, w3], corpus_stats, total_docs, smoothing
            ),
        })
    return genome


# ------------------------------------------------------------------ #
# Public API                                                            #
# ------------------------------------------------------------------ #

def extract_genome(text: str, conn=None, smoothing: float = 0.5) -> list[dict]:
    """Convert a single text into a semantic genome (list of gene dicts)."""
    if not text or not text.strip():
        return []

    tokens, pos_tags = compute_all_pos_tags(text)
    filtered = [
        (t, p) for t, p in zip(tokens, pos_tags)
        if t.strip() and (t.isalpha() or "_" in t)
    ]
    if not filtered:
        return []

    if conn is not None:
        corpus_stats, total_docs = en.fetch_corpus_stats(conn, [t for t, _ in filtered])
    else:
        corpus_stats, total_docs = {}, 0

    return _build_genome_from_parsed(filtered, corpus_stats, total_docs, smoothing)


def extract_genome_batch(
    texts: list[str],
    conn=None,
    smoothing: float = 0.5,
) -> list[list[dict]]:
    """Convert multiple texts into semantic genomes using a single nlp.pipe() call.

    ~3x faster than calling extract_genome() in a loop because spaCy processes
    all texts in one batched pass instead of one pipeline invocation per text.
    Also fetches corpus_stats in a single DB query for all texts combined.
    """
    if not texts:
        return []

    nlp = get_nlp()
    spacy_docs = list(nlp.pipe(texts))

    parsed_list: list[list[tuple[str, str]]] = [
        _filter_spacy_doc(doc) for doc in spacy_docs
    ]

    # One DB round-trip for all tokens across all texts
    all_tokens = [t for pairs in parsed_list for t, _ in pairs]
    if conn is not None and all_tokens:
        corpus_stats, total_docs = en.fetch_corpus_stats(conn, all_tokens)
    else:
        corpus_stats, total_docs = {}, 0

    return [
        _build_genome_from_parsed(pairs, corpus_stats, total_docs, smoothing)
        for pairs in parsed_list
    ]
