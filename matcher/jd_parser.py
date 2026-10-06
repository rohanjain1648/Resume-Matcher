"""Extract requirements from a job description without any skill list.

A requirement is a maximal run of "content" words: words that are not English stopwords and not job-ad filler
("looking", "experience", "years" ...). So
    "building web apps with the MERN stack and REST APIs. Cloud experience is a plus."
gives  web apps | MERN stack | REST APIs | Cloud (nice to have).
"""
import re

from .textutil import FILLER, NICE_CUES, STOP, split_sentences, tokens

YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*[A-Za-z0-9+#]|[A-Za-z0-9]|[,;:()/&]|\.(?=\s|$)")


def _runs(sentence: str):
    """Split one sentence into runs of content words, keeping original casing."""
    runs, cur = [], []
    for raw in WORD_RE.findall(sentence):
        low = raw.lower()
        is_word = raw[0].isalnum()
        if (not is_word) or low in STOP or low in FILLER or re.fullmatch(r"\d+\+?", raw):
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(raw.rstrip("."))
    if cur:
        runs.append(cur)
    return [" ".join(r) for r in runs if len(r) <= 4]


def parse_jd(role: str, jd: str) -> dict:
    reqs = []  # (text, weight, source)
    seen = set()

    def add(text, weight, src):
        key = " ".join(tokens(text))
        if len(key) < 2 or key in seen:
            return
        seen.add(key)
        reqs.append({"text": text, "weight": weight, "source": src})

    for run in _runs(role or ""):
        add(run, 1.5, "role")
    for sent in split_sentences(jd or ""):
        nice = bool(NICE_CUES.search(sent))
        for run in _runs(sent):
            add(run, 0.4 if nice else 1.0, "nice-to-have" if nice else "required")

    years = None
    m = YEARS_RE.search(jd or "")
    if m:
        years = float(m.group(1))
    return {"role": (role or "").strip(), "jd": jd or "", "requirements": reqs, "years_required": years}
