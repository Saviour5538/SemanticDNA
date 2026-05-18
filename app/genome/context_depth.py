import re

FUNCTION_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "which", "what", "whose",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "of", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "whether", "although", "because", "since", "while",
    "if", "unless", "until", "when", "where", "how",
    "i", "me", "we", "us", "you", "he", "him", "she", "they", "them", "it",
    "who", "whom",
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "very", "also", "just", "more", "most", "other", "some", "such", "no",
    "any", "as", "than", "too", "only", "same", "so", "all", "both",
}

COMMON_CONTENT_WORDS = {
    "time", "year", "people", "way", "day", "man", "woman", "child", "work",
    "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "government", "number", "night", "point", "home",
    "water", "room", "mother", "area", "money", "story", "fact", "month",
    "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "among", "while", "name",
    "last", "long", "great", "little", "own", "old", "right", "big", "high",
    "different", "small", "large", "next", "early", "young", "important",
    "public", "private", "real", "best", "free", "able", "put", "take",
    "make", "know", "think", "see", "come", "look", "want", "give", "use",
    "find", "tell", "ask", "seem", "feel", "try", "leave", "call", "keep",
    "let", "begin", "show", "hear", "play", "run", "move", "live", "believe",
    "hold", "bring", "happen", "write", "provide", "sit", "stand", "lose",
    "pay", "meet", "include", "continue", "set", "learn", "change", "lead",
    "understand", "watch", "follow", "stop", "create", "speak", "read",
    "spend", "grow", "open", "walk", "win", "offer", "remember", "consider",
    "appear", "buy", "wait", "serve", "send", "expect", "build", "stay",
    "fall", "cut", "reach", "kill", "remain",
}

_TECHNICAL_PATTERN = re.compile(
    r"(?:[A-Za-z]+\d|\d+[A-Za-z])|(?:^[A-Z]{2,}$)|(?:[-_])"
)

_DOMAIN_SUFFIXES = (
    "tion", "sion", "ness", "ment", "ity", "ism", "ist", "ize", "ise",
    "ify", "able", "ible", "ical", "ous", "ive", "ary", "ery", "ory",
    "logy", "graphy", "metry", "scope", "phile", "phobe",
)


def compute_context_depth(word: str) -> int:
    lower = word.lower()

    if lower in FUNCTION_WORDS:
        return 1

    # L4 — technical markers (check on original word for casing)
    if len(word) > 12:
        return 4
    if _TECHNICAL_PATTERN.search(word):
        return 4

    # L2 — short / common content word
    if len(lower) <= 5 or lower in COMMON_CONTENT_WORDS:
        return 2

    # L3 — domain-specific (has a domain suffix or medium length)
    if any(lower.endswith(sfx) for sfx in _DOMAIN_SUFFIXES):
        return 3

    # Default for medium-length unknown words
    return 3 if len(lower) <= 10 else 4


def compute_depth_for_window(words: list[str]) -> int:
    depths = [compute_context_depth(w) for w in words]
    return round(sum(depths) / len(depths))
