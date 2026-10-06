"""Small text helpers shared by the resume parser, the JD parser and the scorer.

Nothing in here knows about specific skills. The only word lists are generic English stopwords
(from scikit-learn) and a short list of job-ad filler words that are never skills.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = set(ENGLISH_STOP_WORDS)

# Words that appear in job ads / resumes but are not themselves a skill.
FILLER = set("""
looking seeking hiring need needed want wanted candidate candidates applicant role roles position job opening
developer developers engineer engineers executive executives specialist professional professionals
team teams company organization organisation environment
experience experienced experiences year years yr yrs month months
strong good excellent great solid proven demonstrated track record knowledge understanding ability able capable
proficient proficiency familiar familiarity expertise skilled skills skill hands handson hands-on
working work works worked build building built develop developing developed creating create design designing
responsible responsibilities required requirement requirements preferred plus bonus nice have having must
should will would can could also including include includes etc well least minimum relevant similar related
ideal join us our you your we they their using use used via across within across per new
plus advantage desirable optional beneficial
write writes writing wrote run runs running manage managing handle handling perform performing prepare preparing
make making ensure support supporting help helping drive driving deliver delivering
""".split())

# Cues that make a whole JD sentence "nice to have" rather than required.
NICE_CUES = re.compile(
    r"\b(plus|nice[- ]to[- ]have|good[- ]to[- ]have|preferred|bonus|desirable|advantage|optional|beneficial|not required)\b",
    re.I)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")


def canon_token(tok: str) -> str:
    """Normalise one token: node.js / nodejs / node -> 'node'; apis -> 'api'."""
    tok = tok.lower().strip(".")
    tok = tok.replace(".", "")
    if len(tok) > 4 and tok.endswith("js"):
        tok = tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def tokens(text: str):
    return TOKEN_RE.findall(text.lower())


def phrase_key(text: str) -> tuple:
    """Order-preserving tuple of canonical content tokens, used for alias matching."""
    out = []
    for t in tokens(text):
        c = canon_token(t)
        if c and c not in STOP:
            out.append(c)
    return tuple(out)


def is_lookalike(a: tuple, b: tuple) -> bool:
    """Two different single-token names where one is a prefix of the other (java / javascript).

    Embeddings and sub-word tokenisers treat these as close; as skills they are unrelated.
    """
    if len(a) != 1 or len(b) != 1 or a == b:
        return False
    x, y = a[0], b[0]
    return min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x))


def key_overlap(q: tuple, t: tuple) -> float:
    """1.0 if same phrase, 0.85 if the resume term contains every word of the requirement, else 0."""
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    # every word of the requirement is present in the resume term ("REST APIs" in "REST API design");
    # for multi-word phrases a word may also match by prefix ("web apps" ~ "web applications")
    def same(a, b):
        return a == b or (len(q) > 1 and min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))
    return 0.85 if all(any(same(w, x) for x in t) for w in q) else 0.0


def term_regex(term: str):
    """Regex matching `term` as a whole word in free text, tolerant of '.js' / 'js' suffixes."""
    t = term.strip().lower()
    base = re.sub(r"\.?js$", "", t) if len(t) > 4 else t
    suffix_required = base != t
    if " " in base and len(base) > 4 and base.endswith("s") and not base.endswith("ss"):
        base = base[:-1]                      # "web apps" -> "web app"
    pat = re.escape(base).replace(r"\ ", r"[\s\-]+")
    pat = pat + (r"\.?js" if suffix_required else r"(?:\.?js)?")
    # inflection tolerance: plurals, and for multi-word phrases a longer last word ("web app" ~ "web applications")
    pat = pat + (r"[a-z]{0,12}" if " " in base else r"(?:s|es)?")
    return re.compile(r"(?<![a-z0-9+#])" + pat + r"(?![a-z0-9+#])", re.I)


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]
