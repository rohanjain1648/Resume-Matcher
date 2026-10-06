"""Command line interface.

  python cli.py index resumes/            # parse + embed every .txt in a folder (only new/changed ones are redone)
  python cli.py add path/to/new_resume.txt
  python cli.py match --role "MERN Stack Developer" --jd "Looking for ..."   (or --jd-file jd.txt)
"""
import argparse
import json
import pathlib
import sys

from matcher.index import ResumeIndex
from matcher.scoring import match_all


def format_results(out):
    lines = ["Requirements parsed from JD: " + "; ".join(
        f"{r['text']}{' (nice to have)' if r['weight'] < 1 else ''}" for r in out["requirements"])]
    if out["years_required"]:
        lines.append(f"Minimum years asked: {out['years_required']:g}")
    for r in out["results"]:
        lines += ["", f"Rank {r['rank']} · Score {r['score']:.0f} · {r['id']} ({r['headline']})",
                  "Matched: " + (", ".join(r["matched"]) or "-"),
                  "Missing: " + (", ".join(r["missing"]) or "-"),
                  "Why: " + r["why"],
                  *(["Note: near-identical to " + ", ".join(r["near_duplicates"])] if r.get("near_duplicates") else []),
                  f'Evidence: "{r["evidence"]}"',
                  "Score breakdown: " + " | ".join(f"{k}: {v}" for k, v in r["breakdown"].items())]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="index")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index"); s.add_argument("folder")
    s = sub.add_parser("add"); s.add_argument("file")
    s = sub.add_parser("match")
    s.add_argument("--role", default=""); s.add_argument("--jd", default=""); s.add_argument("--jd-file")
    s.add_argument("--json", action="store_true"); s.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    idx = ResumeIndex(a.index_dir)
    if a.cmd == "index":
        print(f"indexed {idx.add_folder(a.folder)} new/changed resumes; {len(idx.docs)} total")
    elif a.cmd == "add":
        print("added" if idx.add_file(a.file) else "unchanged (already indexed)", f"- {len(idx.docs)} resumes total")
    else:
        jd = pathlib.Path(a.jd_file).read_text(encoding="utf8") if a.jd_file else a.jd
        if not (a.role or jd):
            sys.exit("give --role and/or --jd")
        out = match_all(idx, a.role, jd, a.top)
        print(json.dumps(out, indent=2, default=str) if a.json else format_results(out))


if __name__ == "__main__":
    main()
