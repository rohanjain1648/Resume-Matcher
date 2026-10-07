"""Run the 2 evaluation JDs through the matcher and the keyword baseline and compare with eval/expected.md."""
import json
import sys

from matcher import baseline
from matcher.index import ResumeIndex
from matcher.scoring import match_all

JDS = {
    "JD1 MERN": ("MERN Stack Developer",
                 "Looking for a developer with 3+ years of experience building web apps with the MERN stack and REST APIs. "
                 "Cloud experience is a plus.",
                 ["web_01", "web_02", "web_08"]),
    "JD2 Data Analyst": ("Data Analyst",
                         "We need a Data Analyst with 2+ years of experience. You will write SQL, build dashboards in Tableau "
                         "or Power BI and report in Excel. Python is a plus.",
                         ["data_01", "data_03", "data_08"]),
}
idx = ResumeIndex("index")
lines = []
for name, (role, jd, expected) in JDS.items():
    ours = match_all(idx, role, jd, 3)["results"]
    base = baseline.rank(idx, role, jd, 3)
    o_ids, b_ids = [r["id"] for r in ours], [b["id"] for b in base]
    same = lambda ids: {"web_09" if i == "web_08" else i for i in ids}   # web_08 / web_09 are near-identical: either counts
    lines += [f"## {name}", "", f"Expected (written first): {', '.join(expected)}", "",
              "| Rank | Matcher | score | Keyword baseline | score |", "|---|---|---|---|---|"]
    for i in range(3):
        lines.append(f"| {i + 1} | {o_ids[i]} ({ours[i]['headline'][:28]}) | {ours[i]['score']:.0f} | {b_ids[i]} | {base[i]['score']:.0f} |")
    lines += ["", f"Overlap with expected top 3: matcher **{len(same(o_ids) & same(expected))}/3**, "
                  f"baseline **{len(same(b_ids) & same(expected))}/3**", ""]
    for r in ours:
        lines.append(f"- {r['id']}: matched = {'; '.join(r['matched'])} | missing = {'; '.join(r['missing']) or '-'} | {r['why']}")
    lines.append("")
open("eval/results.md", "w", encoding="utf8").write("\n".join(lines))
print("\n".join(lines))
