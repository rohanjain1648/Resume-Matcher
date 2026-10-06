"""Turn raw resume text into the structured facts the matcher uses.

Fairness: the `Name:` line, the education section, college lines and personal-detail lines
(age, gender, DOB ...) are removed here, before anything is embedded or scored.
Nothing downstream ever sees them.
"""
import os
import re
from datetime import date

from .textutil import phrase_key, split_sentences, term_regex

SECTION_NAMES = {
    "summary": "summary", "profile": "summary", "objective": "summary", "about": "summary", "about me": "summary",
    "skills": "skills", "skill": "skills", "technical skills": "skills", "key skills": "skills",
    "core competencies": "skills",
    "experience": "experience", "work experience": "experience", "professional experience": "experience",
    "employment": "experience", "employment history": "experience", "work history": "experience",
    "projects": "projects", "project": "projects",
    "education": "education", "academics": "education", "qualifications": "education",
    "certifications": "other", "certification": "other", "achievements": "other", "languages": "other",
    "interests": "other", "hobbies": "other",
    "personal details": "personal", "personal information": "personal", "personal": "personal",
}
HEADING_RE = re.compile(r"^\s*([A-Za-z ]{3,30}?)\s*:?\s*$")
INLINE_RE = re.compile(r"^\s*([A-Za-z ]{3,25}?)\s*:\s*(\S.*)$")
PERSONAL_RE = re.compile(
    r"^\s*(age|dob|date of birth|gender|sex|marital status|father'?s? name|nationality|religion|college|university|"
    r"school|address|photo)\b.*$", re.I)

# 'Present' on a resume is resolved against this date so scores are reproducible. The sample resumes were written
# around the start of 2025; override with MATCHER_AS_OF=YYYY-MM-DD (or 'today').
_env = os.environ.get("MATCHER_AS_OF", "2025-01-01")
DEFAULT_AS_OF = date.today() if _env == "today" else date.fromisoformat(_env)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
DATE_RANGE_RE = re.compile(
    r"(?:(?P<m1>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<y1>(?:19|20)\d{2})\s*(?:-|–|—|to)\s*"
    r"(?:(?P<now>present|current|now|till date|date)|(?:(?P<m2>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<y2>(?:19|20)\d{2}))",
    re.I)
JOB_HEADER_RE = re.compile(
    r"^(?P<title>[^(\n,]+?)\s*(?:,|\s[-–—|@]\s|\s+at\s)\s*(?P<company>[^(\n]+?)\s*\((?P<dates>[^)]*(?:19|20)\d{2}[^)]*)\)\s*:?$")
STATED_YEARS_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.I)
# code-looking tokens: Node.js, MongoDB, C++, AWS, SQL ... (no digits, to skip things like FY23)
MINED_RE = re.compile(r"\b(?:[A-Za-z]+\.js|[A-Z][a-z]+[A-Z][A-Za-z]*|[A-Z]{2,}[a-z]*|[A-Za-z]+[+#]+)\b")
NOISE_MINED = {"B.TECH", "MBA", "BBA", "BCA", "MCA", "CSE", "IIT", "NIT", "VIT"}


def _months_between(y1, m1, y2, m2):
    return max(0, (y2 - y1) * 12 + (m2 - m1) + 1)


def _month(tok, default):
    return MONTHS.get((tok or "").lower()[:3], default)


def years_from_dates(text: str, as_of: date):
    """Union of all 'Mon YYYY - Mon YYYY/Present' ranges, in years. Overlaps are not double counted."""
    spans = []
    for m in DATE_RANGE_RE.finditer(text):
        y1, mo1 = int(m["y1"]), _month(m["m1"], 1)
        if m["now"]:
            y2, mo2 = as_of.year, as_of.month
        else:
            y2, mo2 = int(m["y2"]), _month(m["m2"], 12)
        if (y2, mo2) > (as_of.year, as_of.month):
            y2, mo2 = as_of.year, as_of.month
        if (y2, mo2) >= (y1, mo1):
            spans.append((y1 * 12 + mo1, y2 * 12 + mo2))
    spans.sort()
    total, cur = 0, None
    for s, e in spans:
        if cur is None:
            cur = [s, e]
        elif s <= cur[1] + 1:
            cur[1] = max(cur[1], e)
        else:
            total += cur[1] - cur[0] + 1
            cur = [s, e]
    if cur:
        total += cur[1] - cur[0] + 1
    return total / 12.0, len(spans)


def split_sections(text: str):
    """Return (headline, {section: [lines]}). Unlabelled leading text goes to 'summary'."""
    headline = ""
    sections = {}
    current = "summary"
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            sections.setdefault(current, []).append("")
            continue
        low = line.strip().lower()
        if low.startswith("name:"):
            continue  # never used
        if low.startswith("headline:"):
            headline = line.split(":", 1)[1].strip()
            continue
        if PERSONAL_RE.match(line):
            continue
        h = HEADING_RE.match(line)
        if h and h.group(1).strip().lower() in SECTION_NAMES:
            current = SECTION_NAMES[h.group(1).strip().lower()]
            continue
        i = INLINE_RE.match(line)
        if i and i.group(1).strip().lower() in SECTION_NAMES:
            sec = SECTION_NAMES[i.group(1).strip().lower()]
            sections.setdefault(sec, []).append(i.group(2).strip())
            continue
        sections.setdefault(current, []).append(line.strip())
    sections.pop("education", None)
    sections.pop("personal", None)
    return headline, sections


def _split_skill_items(lines):
    items = []
    for line in lines:
        for chunk in re.split(r"[,;|•\n](?![^(]*\))|\s{2,}", line):
            chunk = chunk.strip(" -*•\t.")
            if not chunk:
                continue
            inner = re.findall(r"\(([^)]*)\)", chunk)
            outer = re.sub(r"\([^)]*\)", "", chunk).strip(" -")
            if outer:
                items.append(outer)
            for grp in inner:
                for sub in re.split(r"[,/]", grp):
                    if sub.strip():
                        items.append(sub.strip())
    out, seen = [], set()
    for it in items:
        if len(it.split()) > 4 or len(it) < 2:
            continue
        k = phrase_key(it)
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def parse_resume(text: str, as_of: date = None) -> dict:
    as_of = as_of or DEFAULT_AS_OF
    headline, sec = split_sections(text)
    skills_lines = sec.get("skills", [])
    skill_terms = _split_skill_items(skills_lines)

    exp_lines = sec.get("experience", []) + sec.get("projects", [])
    summary_lines = sec.get("summary", [])
    if not any(l.strip() for l in sec.get("experience", [])) and not sec.get("projects"):
        exp_lines = list(summary_lines)  # free-form resume: the body is the experience
    exp_text = "\n".join(exp_lines)

    # --- evidence sentences ------------------------------------------------------------------
    bullets, titles = [], []
    if headline:
        titles.append(headline)
    for line in exp_lines + summary_lines:
        s = line.strip()
        if not s:
            continue
        hm = JOB_HEADER_RE.match(s)
        if hm:
            titles.append(hm["title"].strip())
            continue
        s = s.lstrip("-*• ").strip()
        for sent in (split_sentences(s) if len(s) > 160 else [s]):
            if len(sent.split()) >= 3 and not DATE_RANGE_RE.fullmatch(sent):
                bullets.append(sent)
    bullets = list(dict.fromkeys(bullets))

    # --- terms: listed skills + code-looking tokens mined from experience -------------------------
    mined = []
    for m in MINED_RE.finditer(exp_text):
        w = m.group(0)
        if w.upper() not in NOISE_MINED:
            mined.append(w)
    terms, seen = [], set()
    for t in skill_terms + mined:
        k = phrase_key(t)
        if k and k not in seen:
            seen.add(k)
            terms.append(t)

    demonstrated = {}
    for t in terms:
        demonstrated[t] = bool(term_regex(t).search(exp_text))

    # --- years of experience -----------------------------------------------------------------
    yrs_dates, n_spans = years_from_dates(exp_text + "\n" + "\n".join(summary_lines), as_of)
    stated = None
    m = STATED_YEARS_RE.search("\n".join(summary_lines) + " " + headline)
    if m:
        stated = float(m.group(1))
    if n_spans:
        years, years_src = yrs_dates, f"{n_spans} dated job(s)"
    elif stated is not None:
        years, years_src = stated, "stated in summary"
    else:
        years, years_src = 0.0, "no dated experience"

    return {
        "headline": headline,
        "titles": list(dict.fromkeys(titles)),
        "skill_terms": skill_terms,
        "terms": terms,
        "demonstrated": demonstrated,
        "bullets": bullets,
        "skills_line": ", ".join(skill_terms),
        "exp_text": exp_text,
        "summary_text": " ".join(summary_lines),
        "years": round(years, 2),
        "years_src": years_src,
    }
