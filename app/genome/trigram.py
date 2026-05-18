import hashlib

_PAD = "__PAD__"


def extract_trigram_windows(tokens: list[str]) -> list[tuple[str, str, str]]:
    n = len(tokens)
    if n == 0:
        return []
    if n == 1:
        return [(_PAD, tokens[0], _PAD)]
    if n == 2:
        return [(tokens[0], tokens[1], _PAD)]
    return [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(n - 2)]


def hash_trigram(w1: str, w2: str, w3: str) -> str:
    phrase = f"{w1.lower()}|{w2.lower()}|{w3.lower()}"
    return hashlib.md5(phrase.encode()).hexdigest()[:12]


def compute_trigram_hashes(tokens: list[str]) -> list[str]:
    return [hash_trigram(w1, w2, w3) for w1, w2, w3 in extract_trigram_windows(tokens)]
