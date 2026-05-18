import re


def compute_phonetic_code(word: str) -> str:
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return "__VOW__"

    # Consonant normalization — order matters
    w = w.replace("ck", "k")
    w = w.replace("ph", "f")
    w = w.replace("gh", "f")
    w = w.replace("wh", "w")
    w = w.replace("qu", "kw")
    w = w.replace("x", "ks")

    # Soft-c before e/i/y → s; remaining c → k
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k")

    w = w.replace("z", "s")

    # Collapse consecutive duplicate consonants
    for ch in "nlmpstr":
        w = re.sub(rf"{ch}{{2,}}", ch, w)

    # Remove vowels
    w = re.sub(r"[aeiou]", "", w)

    return w.upper() if w else "__VOW__"


def compute_phonetic_sequence(words: list[str]) -> str:
    return "-".join(compute_phonetic_code(w) for w in words)
