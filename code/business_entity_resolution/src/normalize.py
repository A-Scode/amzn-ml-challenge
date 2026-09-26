"""
Normalization utilities for business names and addresses.

Everything here is deterministic, rule-based, and self-contained (no network
calls, no external lookups) — safe under the challenge's fair-play rules.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Legal-suffix / abbreviation canonicalization for business names
# ---------------------------------------------------------------------------
# Map noisy variants -> a single canonical short token. Longest keys first so
# multi-word variants ("private limited") match before single words.
_NAME_REPLACEMENTS = [
    (r"\bprivate limited\b", "pvt ltd"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\blimited liability company\b", "llc"),
    (r"\bcorporation\b", "corp"),
    (r"\bincorporated\b", "inc"),
    (r"\bcompany\b", "co"),
    (r"\bprivate\b", "pvt"),
    (r"\blimited\b", "ltd"),
    (r"\bpublic ltd\b", "ltd"),
    (r"\bsociete a responsabilite limitee\b", "sarl"),
    (r"\bsociete par actions simplifiee\b", "sas"),
    (r"\bsociete anonyme\b", "sa"),
    (r"&", " and "),
]

# Tokens that carry almost no discriminative value for blocking purposes.
# NOTE: these are still used as *features* (suffix-match signal), just not as
# blocking keys, since they're too generic (huge candidate blocks).
GENERIC_NAME_TOKENS = {
    "the", "and", "of", "a", "an", "inc", "co", "corp", "ltd", "llc", "llp",
    "pvt", "sarl", "sas", "sa", "gmbh", "plc", "group", "groupe", "enterprises",
    "enterprise", "trading", "services", "service", "solutions", "consultants",
    "consulting", "international", "india", "national", "national",
}

GENERIC_ADDR_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave", "near", "no", "door",
    "block", "colony", "nagar", "cross", "main", "lane", "ln", "floor",
    "unit", "apartment", "apt", "phase", "sector", "industrial", "area",
    "po", "box", "county", "district", "dist", "null", "rue", "boulevard",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_POSTAL_RE = re.compile(r"\b\d{4,6}\b")

# ---------------------------------------------------------------------------
# Minimal Devanagari -> Latin phonetic transliteration.
# A plain character substitution table (like a codec) — deterministic and
# self-contained, used only to give the blocker/feature layer a fighting
# chance when the same business appears in Devanagari in one source and in
# Latin script (transliterated) in another. This is NOT a lookup against any
# external business/identity database.
# ---------------------------------------------------------------------------
_DEVANAGARI_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "अं": "an", "अः": "ah",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "ः": "h", "्": "",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9",
}


def transliterate_devanagari(text: str) -> str:
    """Best-effort char-by-char Devanagari -> Latin phonetic transliteration."""
    out = []
    for ch in text:
        if "\u0900" <= ch <= "\u097F":
            out.append(_DEVANAGARI_MAP.get(ch, ""))
        else:
            out.append(ch)
    return "".join(out)


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def strip_accents(text: str) -> str:
    """Fold accented Latin characters (Cyrillic/Latin-Extended) to ASCII-ish."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.strip().lower() in ("nan", "null", "none", ""):
        return ""
    if has_devanagari(text):
        text = transliterate_devanagari(text)
    text = strip_accents(text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(raw_name: str):
    """Return dict with cleaned name variants used for features/blocking."""
    clean = basic_clean(raw_name)
    for pattern, repl in _NAME_REPLACEMENTS:
        clean = re.sub(pattern, repl, clean)
    clean = _WS_RE.sub(" ", clean).strip()
    tokens = clean.split()
    core_tokens = [t for t in tokens if t not in GENERIC_NAME_TOKENS]
    core = " ".join(core_tokens) if core_tokens else clean
    acronym = "".join(t[0] for t in tokens if t) if tokens else ""
    return {
        "clean": clean,
        "core": core,
        "tokens": tokens,
        "core_tokens": core_tokens,
        "first_token": core_tokens[0] if core_tokens else (tokens[0] if tokens else ""),
        "acronym": acronym,
    }


def normalize_address(raw_addr: str):
    clean = basic_clean(raw_addr)
    postal_match = _POSTAL_RE.findall(clean)
    postal_code = postal_match[0] if postal_match else ""
    tokens = clean.split()
    core_tokens = [t for t in tokens if t not in GENERIC_ADDR_TOKENS and not t.isdigit()]
    return {
        "clean": clean,
        "tokens": tokens,
        "core_tokens": core_tokens,
        "postal_code": postal_code,
    }
