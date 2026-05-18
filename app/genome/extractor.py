from app.genome import trigram as tg
from app.genome import phonetic as ph
from app.genome import context_depth as cd
from app.genome import entropy as en
from app.genome.pos_tagger import compute_all_pos_tags

_PAD = "__PAD__"


def extract_genome(text: str, conn=None, smoothing: float = 0.5) -> list[dict]:
    """Convert raw text into a semantic genome (list of gene dicts).

    Each gene corresponds to one 3-word sliding window and encodes:
    trigram_hash, phonetic_code, pos_sequence, context_depth, entropy_score.
    """
    if not text or not text.strip():
        return []

    tokens, pos_tags = compute_all_pos_tags(text)

    # Remove pure whitespace/punctuation tokens; keep alpha words and __PAD__
    filtered: list[tuple[str, str]] = [
        (t, p) for t, p in zip(tokens, pos_tags) if t.strip() and (t.isalpha() or "_" in t)
    ]

    if not filtered:
        return []

    tokens = [t for t, _ in filtered]
    pos_tags = [p for _, p in filtered]

    # Fetch IDF stats (no-op if no DB connection — entropy will be 0)
    if conn is not None:
        corpus_stats, total_docs = en.fetch_corpus_stats(conn, tokens)
    else:
        corpus_stats, total_docs = {}, 0

    # Pad so we always have at least one window
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

        gene = {
            "trigram_hash": tg.hash_trigram(w1, w2, w3),
            "phonetic_code": ph.compute_phonetic_sequence([w1, w2, w3]),
            "pos_sequence": f"{p1}-{p2}-{p3}",
            "context_depth": cd.compute_depth_for_window([w1, w2, w3]),
            "entropy_score": en.compute_entropy_for_window(
                [w1, w2, w3], corpus_stats, total_docs, smoothing
            ),
        }
        genome.append(gene)

    return genome
