"""Score every indexed resume against a JD and explain each number.

Score (0-100) = 55% skill coverage + 15% experience fit + 15% title fit + 15% overall semantic fit.

Skill coverage, per JD requirement, takes the strongest of:
  * exact / alias match with a skill the resume lists or mentions   -> 1.0   (x0.55 if only listed, never used)
  * related skills, found by embedding similarity that is standardised per requirement (z-score against every
    skill in the index), so "closer than usual" rather than a global magic cosine. Capped at REL_MAX.
  * a resume sentence that is unusually similar to the requirement   -> capped at SENT_MAX
Look-alike names (java / javascript) get no credit from similarity.
"""
import re

import numpy as np

from . import embedder, knowledge
from .jd_parser import parse_jd
from .textutil import is_lookalike, key_overlap, phrase_key, term_regex

W_SKILL, W_EXP, W_TITLE, W_HOLISTIC = 0.55, 0.15, 0.15, 0.15
LISTED_FACTOR = 0.55      # skill only appears in the skills list, never in experience
REL_MAX, SENT_MAX = 0.5, 0.5
TERM_Z = (2.0, 3.5)       # z-score range mapped to 0..1 for related-skill similarity
SENT_Z = (2.0, 4.0)
MIN_TERM_COS = 0.5
MENTION = 0.3             # strength contributed by each resume skill named in the requirement's definition
DUP_COS = 0.97           # resume-to-resume similarity above which two resumes are flagged as near-duplicates
MATCH_AT = 0.35           # strength needed to count as "matched"


def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _soft_or(vals):
    p = 1.0
    for v in vals:
        p *= (1.0 - v)
    return 1.0 - p


def _stats(sims):
    return float(np.mean(sims)), float(np.std(sims)) + 1e-6


def _z_over(background, vec, exclude=None):
    sims = background @ vec
    if exclude is not None and len(sims) > 1:
        keep = np.ones(len(sims), bool)
        keep[exclude] = False
        mu, sd = _stats(sims[keep])
    else:
        mu, sd = _stats(sims)
    return mu, sd


def match_all(index, role: str, jd: str, top_k: int = 5):
    spec = parse_jd(role, jd)
    reqs = spec["requirements"]
    ctx = (role + ". " + jd)[:300]

    def scorer(blobs, query):
        return embedder.encode(blobs) @ embedder.encode([query])[0]

    for r in reqs:
        r["know"] = knowledge.expand(r["text"], ctx, scorer)
    plain = embedder.encode([r["text"] for r in reqs])
    rich = embedder.encode([r["text"] + ". " + r["know"][:400] if r["know"] else r["text"] for r in reqs])
    req_vecs = plain + rich
    req_vecs = req_vecs / np.maximum(np.linalg.norm(req_vecs, axis=1, keepdims=True), 1e-9)
    jd_vec = embedder.encode([(role + ". " + jd)[:2000]])[0]
    role_vec = embedder.encode([role or jd[:100]])[0]
    for r, v in zip(reqs, req_vecs):
        r["key"], r["vec"] = phrase_key(r["text"]), v
        if len(index.vocab_keys) >= 20:
            excl = [i for i, k in enumerate(index.vocab_keys) if k == r["key"]] or None
            r["tmu"], r["tsd"] = _z_over(index.vocab, v, excl)
        else:
            r["tmu"] = r["tsd"] = None
        if len(index.bullet_bg) >= 20:
            r["bmu"], r["bsd"] = _z_over(index.bullet_bg, v)
        else:
            r["bmu"] = r["bsd"] = None
    t_mu, t_sd = _z_over(index.title_bg, role_vec) if len(index.title_bg) >= 5 else (None, None)
    h_mu, h_sd = _z_over(index.doc_bg, jd_vec) if len(index.doc_bg) >= 5 else (None, None)

    # pass 1: how common is each requirement in the collection? (a requirement every resume satisfies says little)
    n_docs = max(1, len(index.docs))
    df = [0] * len(reqs)
    for d in index.docs.values():
        for i, b in enumerate(_score_one(d, spec, reqs, jd_vec, role_vec, None, None, strengths_only=True)):
            df[i] += b["strength"] >= 0.5
    for r, c in zip(reqs, df):
        specificity = np.log((n_docs + 1) / (c + 0.5)) / np.log(n_docs + 1)
        r["idf"] = 0.4 + 0.6 * float(np.clip(specificity, 0, 1))
        r["df"] = c
    # pass 2: full scoring with specificity-adjusted weights
    results = [_score_one(d, spec, reqs, jd_vec, role_vec, (t_mu, t_sd), (h_mu, h_sd)) for d in index.docs.values()]
    results.sort(key=lambda r: (-r["score"], r["id"]))
    for i, r in enumerate(results, 1):
        r["rank"] = i
    vec = {d["id"]: d["doc_vec"] for d in index.docs.values()}
    for r in results[:top_k]:      # flag near-identical resumes so a recruiter is not shown the same person twice
        dups = [o["id"] for o in results if o["id"] != r["id"] and float(vec[o["id"]] @ vec[r["id"]]) >= DUP_COS
                and abs(o["score"] - r["score"]) < 5]
        r["near_duplicates"] = dups
    return {"requirements": [{"text": r["text"], "weight": r["weight"], "source": r["source"],
                              "specificity": round(r["idf"], 2), "resumes_matching": r["df"]} for r in reqs],
            "years_required": spec["years_required"], "results": results[:top_k], "n_candidates": len(results)}


def _term_z(r, cos):
    if r["tmu"] is None:
        return (cos - 0.6) / 0.07
    return (cos - r["tmu"]) / r["tsd"]


def _score_one(d, spec, reqs, jd_vec, role_vec, t_stats, h_stats, strengths_only=False):
    p = d["parsed"]
    terms, tvecs, bvecs = p["terms"], d["term_vecs"], d["bullet_vecs"]
    tkeys = [phrase_key(t) for t in terms]
    exp_text = p.get("exp_text", "")
    per_req, matched_terms = [], set()

    for r in reqs:
        rk, rre = r["key"], term_regex(r["text"])
        best = {"strength": 0.0, "kind": "none", "terms": [], "listed_only": False}

        # 1) exact / alias matches among the resume's skill terms
        for t, tk in zip(terms, tkeys):
            ov = key_overlap(rk, tk)
            if ov:
                demo = p["demonstrated"].get(t, False)
                s = ov * (1.0 if demo else LISTED_FACTOR)
                if s > best["strength"]:
                    best = {"strength": s, "kind": "exact", "terms": [t], "listed_only": not demo}
        # ... or the wording appears in the experience text itself
        if best["kind"] != "exact" and rre.search(exp_text):
            best = {"strength": 1.0, "kind": "exact", "terms": [r["text"]], "listed_only": False}
        elif best["strength"] < 0.7 and rre.search(p.get("summary_text", "")):   # only claimed in the summary
            best = {"strength": 0.7, "kind": "exact", "terms": [r["text"]], "listed_only": False}

        # 2) related skills by (standardised) embedding similarity
        rel = []
        if len(tvecs) and best["strength"] < 1.0:
            cos_all = tvecs @ r["vec"]
            for i, t in enumerate(terms):
                if key_overlap(rk, tkeys[i]) or is_lookalike(rk, tkeys[i]) or tkeys[i] == rk:
                    continue
                if cos_all[i] < MIN_TERM_COS:
                    continue
                s = REL_MAX * ramp(_term_z(r, float(cos_all[i])), *TERM_Z)
                if s > 0:
                    if not p["demonstrated"].get(t, False):
                        s *= LISTED_FACTOR
                    rel.append((s, t))
            rel.sort(reverse=True)
        rel_s = _soft_or([s for s, _ in rel[:3]])
        # skills that the looked-up definition of the requirement names (MERN -> MongoDB, Express, React, Node.js)
        named = []
        if r["know"] and best["strength"] < 1.0:
            for i, t in enumerate(terms):
                if len(t) > 2 and not (key_overlap(rk, tkeys[i]) or is_lookalike(rk, tkeys[i]) or tkeys[i] == rk)                         and term_regex(t).search(r["know"]):
                    named.append((MENTION * (1.0 if p["demonstrated"].get(t, False) else LISTED_FACTOR), t))
        if named:
            named.sort(reverse=True)
            ms = min(0.8, _soft_or([s for s, _ in named]))
            if ms > rel_s:
                rel_s, rel = ms, named
        rel_s = min(rel_s, 0.8)

        # 3) a sentence of the resume that is unusually close to the requirement
        sent_s, sent_i = 0.0, None
        if len(bvecs):
            cb = bvecs @ r["vec"]
            sent_i = int(np.argmax(cb))
            z = (float(cb[sent_i]) - r["bmu"]) / r["bsd"] if r["bmu"] is not None else (float(cb[sent_i]) - 0.55) / 0.06
            sent_s = SENT_MAX * ramp(z, *SENT_Z)

        sem = min(0.8, _soft_or([rel_s, sent_s]))
        if sem > best["strength"]:
            best = {"strength": sem, "kind": "related", "terms": [t for _, t in rel[:3]] if rel_s >= sent_s else [],
                    "listed_only": False, "sentence": sent_i if sent_s > rel_s else None}
        best["req"] = r
        per_req.append(best)
        if best["strength"] >= MATCH_AT:
            matched_terms.update(best["terms"])

    if strengths_only:
        return per_req
    ws = [b["req"]["weight"] * b["req"]["idf"] for b in per_req]
    wsum = sum(ws) or 1.0
    skill_cov = sum(b["strength"] * w for b, w in zip(per_req, ws)) / wsum

    # experience
    yrs, need = p["years"], spec["years_required"]
    if need:
        exp_fit = min(1.0, yrs / need)
        exp_reason = f"{yrs:g} years of dated experience vs {need:g}+ required"
    else:
        exp_fit = 0.5 + 0.5 * min(1.0, yrs / 3.0)
        exp_reason = f"{yrs:g} years of dated experience (JD states no minimum)"

    # titles
    if len(d["title_vecs"]):
        ct = d["title_vecs"] @ role_vec
        bi = int(np.argmax(ct))
        zt = (float(ct[bi]) - t_stats[0]) / t_stats[1] if t_stats[0] is not None else (float(ct[bi]) - 0.6) / 0.08
        title_fit = ramp(zt, 0.5, 3.0)
        best_title = p["titles"][bi]
    else:
        title_fit, best_title = 0.0, "none found"

    # overall semantic fit of the whole resume vs the whole JD
    ch = float(d["doc_vec"] @ jd_vec)
    zh = (ch - h_stats[0]) / h_stats[1] if h_stats[0] is not None else (ch - 0.6) / 0.07
    holistic = ramp(zh, 0.0, 2.5)

    score = 100.0 * (W_SKILL * skill_cov + W_EXP * exp_fit + W_TITLE * title_fit + W_HOLISTIC * holistic)
    shortfall = 1.0
    if need and yrs < need:      # below the stated minimum: scale the whole score, not just the 15% experience part
        shortfall = 0.6 + 0.4 * exp_fit
        score *= shortfall

    matched, missing, listed_only = [], [], []
    for b in per_req:
        r = b["req"]
        if b["strength"] >= MATCH_AT:
            if b["kind"] == "exact":
                label = r["text"] if b["terms"][0].lower() == r["text"].lower() else f"{b['terms'][0]} (matches {r['text']})"
                if b["listed_only"]:
                    label += " [listed in skills, not shown in any job]"
                    listed_only.append(r["text"])
            elif b["terms"]:
                label = f"{', '.join(b['terms'])} (counts as {r['text']})"
            else:
                label = f"{r['text']} (similar work in: \"{p['bullets'][b['sentence']][:60]}...\")"
            matched.append(label)
        else:
            missing.append(r["text"] + (" (weak evidence)" if b["strength"] >= 0.1 else " (not mentioned)"))

    evidence = _pick_evidence(p, per_req, jd_vec, d["bullet_vecs"])
    why = _why(p, per_req, yrs, need, matched, missing, listed_only)
    return {
        "id": d["id"], "name": _display_name(d["text"]), "headline": p["headline"], "score": round(score, 1),
        "matched": matched, "missing": missing, "why": why, "evidence": evidence,
        "breakdown": {
            "skill_coverage": f"{skill_cov * 100:.0f}% (weight {W_SKILL:.0%})",
            "experience": f"{exp_fit * 100:.0f}% - {exp_reason} (weight {W_EXP:.0%})" + (f"; total score also x{shortfall:.2f} for being under the minimum" if shortfall < 1 else ""),
            "title": f"{title_fit * 100:.0f}% - closest title '{best_title}' (weight {W_TITLE:.0%})",
            "overall_similarity": f"{holistic * 100:.0f}% (weight {W_HOLISTIC:.0%})",
        },
        "years": yrs,
    }


def _display_name(text):
    m = re.search(r"(?im)^name:\s*(.+)$", text)
    return m.group(1).strip() if m else ""   # display only; never used for scoring


def _pick_evidence(p, per_req, jd_vec, bvecs):
    bullets = p["bullets"]
    if not bullets:
        return ""
    hits = np.zeros(len(bullets))
    for b in per_req:
        if b["strength"] < MATCH_AT:
            continue
        pats = [term_regex(t) for t in b["terms"]]
        for i, s in enumerate(bullets):
            if any(pt.search(s) for pt in pats):
                hits[i] += b["strength"] * b["req"]["weight"]
        if b.get("sentence") is not None:
            hits[b["sentence"]] += 0.5 * b["strength"]
    sim = bvecs @ jd_vec
    i = int(np.argmax(hits + 0.01 * sim))
    return bullets[i]


def _why(p, per_req, yrs, need, matched, missing, listed_only):
    n = len(per_req)
    ex = sum(1 for b in per_req if b["kind"] == "exact" and b["strength"] >= MATCH_AT and not b["listed_only"])
    rel = sum(1 for b in per_req if b["kind"] == "related" and b["strength"] >= MATCH_AT)
    parts = []
    if yrs >= 1:
        parts.append(f"{yrs:.0f} years' dated experience" + (f" (asked {need:g}+)" if need else ""))
    else:
        parts.append("no/under 1 year of dated experience")
    parts.append(f"{ex} of {n} requirements shown directly in work history" + (f", {rel} via related skills" if rel else ""))
    if listed_only:
        parts.append(f"{len(listed_only)} only listed in the skills section")
    if missing:
        parts.append(f"missing: {', '.join(m.split(' (')[0] for m in missing[:3])}")
    return "; ".join(parts) + "."
