"""Optional knowledge expansion for JD requirements.

Small embedding models do not know that "MERN" means MongoDB + Express + React + Node.js. So each requirement is
looked up once in Wikipedia (intro paragraph of the best search hit) and cached in knowledge_cache.json.
The definition text is then used to (a) find resume skills the definition mentions and (b) enrich the embedding.
If the network is unavailable the lookup returns "" and the matcher falls back to embeddings alone.
No skill pairs are written anywhere in this repo; the cache only holds what Wikipedia returned.
"""
import json
import os
import pathlib
import re
import urllib.parse
import urllib.request

CACHE_PATH = pathlib.Path(os.environ.get("MATCHER_KNOWLEDGE_CACHE", "knowledge_cache.json"))
ENABLED = os.environ.get("MATCHER_KNOWLEDGE", "1") != "0"
_cache = None


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf8"))
        except Exception:
            _cache = {}
    return _cache


def _api(**params):
    params["format"] = "json"
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "resume-matcher-assignment/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.load(r)


def _fetch(term: str, context: str = "", scorer=None) -> str:
    hits = _api(action="query", list="search", srsearch=term, srlimit=5)["query"]["search"]
    toks = [t[:4] for t in re.findall(r"[a-z0-9]+", term.lower())]
    cands = []
    for h in hits:                       # hits whose title/snippet mention every word of the term
        blob = h["title"] + ". " + re.sub(r"<[^>]+>", "", h["snippet"])
        if all(t in blob.lower() for t in toks):
            cands.append((h, blob))
    if not cands:
        return ""
    best = cands[0][0]
    if scorer and context and len(cands) > 1:   # ambiguous term ("cloud"): pick the sense closest to the JD
        sc = scorer([b for _, b in cands], term + ". " + context)
        best = cands[int(max(range(len(cands)), key=lambda i: sc[i]))][0]
    pages = _api(action="query", pageids=best["pageid"], prop="extracts", exintro=1, explaintext=1,
                 exchars=1200)["query"]["pages"]
    return " ".join(p.get("extract", "") for p in pages.values()).strip()


def expand(term: str, context: str = "", scorer=None) -> str:
    if not ENABLED:
        return ""
    cache = _load()
    key = term.strip().lower() + ("|" + context[:60].lower() if context else "")
    if key in cache:
        return cache[key]
    try:
        text = _fetch(term, context, scorer)
    except Exception:
        return ""          # offline: do not cache failures
    cache[key] = text
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf8")
    except Exception:
        pass
    return text
