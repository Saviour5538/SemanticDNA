import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _nlp


def compute_all_pos_tags(text: str) -> tuple[list[str], list[str]]:
    """Return (tokens, pos_tags) aligned lists for the given text."""
    nlp = get_nlp()
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_space]
    pos_tags = [token.pos_ for token in doc if not token.is_space]
    return tokens, pos_tags


def pos_sequence_similarity(seq1: str, seq2: str) -> float:
    """Position-wise POS tag match ratio for two hyphen-separated sequences."""
    tags1 = seq1.split("-")
    tags2 = seq2.split("-")
    if not tags1 or not tags2:
        return 0.0
    length = max(len(tags1), len(tags2))
    matches = sum(1 for a, b in zip(tags1, tags2) if a == b)
    return matches / length
