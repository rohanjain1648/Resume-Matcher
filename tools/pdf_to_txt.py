"""Convert PDF resumes into the plain-text format the matcher indexes.

    python tools/pdf_to_txt.py resumes/*.pdf --out resumes --prefix manual

For each PDF it writes <prefix>_NN.txt containing a `Name:` line (first name-looking line, ignored by the matcher)
followed by the text with contact details (emails, phones, links) removed. A PDF that has none of the usual resume
sections (a certificate or letter, say) is skipped with a warning. The original PDFs are left untouched.
"""
import argparse
import pathlib
import re

import pymupdf

EMAIL = re.compile(r"\S+@\S+\.\S+")
PHONE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")
LINK = re.compile(r"(https?://\S+|www\.\S+|\b(?:linkedin|github)\.com/\S*|\b[\w\-]+(?:\.[\w\-]+)*\.(?:app|com|io|dev|me|in|net|org)\b\S*)", re.I)
BULLET_CHARS = "•●▪◦■‣"
SECTION_WORDS = re.compile(r"^\s*(experience|work experience|skills|technical skills|education|projects)\s*:?\s*$", re.I)


def clean(raw: str):
    lines, name = [], None
    for ln in raw.splitlines():
        ln = ln.replace(" ", " ").strip()
        if not ln:
            continue
        if ln[0] in BULLET_CHARS:
            ln = "- " + ln[1:].strip()
        ln = LINK.sub("", EMAIL.sub("", PHONE.sub("", ln))).strip(" |#●" + BULLET_CHARS)
        if not ln or re.fullmatch(r"(?:(?:leetcode|codechef|codeforces|github|linkedin|baylore|portfolio|twitter|youtube|link|code link)\W*)+\S{0,40}", ln, re.I):
            continue          # a bare "LinkedIn / GitHub / LeetCode" line (the URL itself is gone)
        if name is None and re.fullmatch(r"[A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){1,3}", ln) and not SECTION_WORDS.match(ln):
            name = ln
            continue
        if re.match(r"last updated", ln, re.I):
            continue
        lines.append(ln)
    return name, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", default="resumes")
    ap.add_argument("--prefix", default="manual")
    a = ap.parse_args()
    n = 0
    for p in sorted(a.pdfs):
        doc = pymupdf.open(p)
        name, lines = clean("\n".join(pg.get_text() for pg in doc))
        if not any(SECTION_WORDS.match(l) for l in lines):
            print(f"SKIPPED {p}: no resume sections found (not a resume?)")
            continue
        n += 1
        out = pathlib.Path(a.out) / f"{a.prefix}_{n:02d}.txt"
        out.write_text(f"Name: {name or 'Unknown'}\n\n" + "\n".join(lines) + "\n", encoding="utf8")
        print(f"{p} -> {out}")


if __name__ == "__main__":
    main()
