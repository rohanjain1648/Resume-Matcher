"""Plain keyword-search baseline: count how many distinct JD words appear verbatim in the resume."""
import re

from .resume_parser import split_sections
from .textutil import FILLER, STOP, tokens


def keywords(text):
    return {t for t in tokens(text) if t not in STOP and t not in FILLER and not re.fullmatch(r"\d+\+?", t)}


def rank(index, role, jd, top_k=3):
    kws = keywords(role + " " + jd)
    out = []
    for d in index.docs.values():
        _, sec = split_sections(d["text"])           # same name/education stripping as the main system
        body = " ".join(" ".join(v) for v in sec.values())
        words = set(tokens(body))
        hit = sorted(k for k in kws if k in words)
        out.append({"id": d["id"], "score": round(100 * len(hit) / max(1, len(kws)), 1), "matched": hit})
    out.sort(key=lambda r: (-r["score"], r["id"]))
    return out[:top_k]
