"""python checks/test_bias_and_incremental.py

1. Bias: the same resume with a different name / college / gender / age line must get exactly the same score.
2. Incremental: adding one resume parses + embeds only that file and leaves every other pickle untouched.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import shutil
import tempfile

from matcher.index import ResumeIndex
from matcher.scoring import match_all

ROLE = "MERN Stack Developer"
JD = "Looking for a developer with 3+ years of experience building web apps with the MERN stack and REST APIs. Cloud experience is a plus."

tmp = pathlib.Path(tempfile.mkdtemp())
shutil.copytree("index", tmp / "index")
idx = ResumeIndex(str(tmp / "index"))

# ---- 1. bias ---------------------------------------------------------------------------------------------
orig = pathlib.Path("resumes/web_02.txt").read_text(encoding="utf8")
variant = (orig.replace("Neha Kapoor", "Mohammed Al-Rashid").replace("Delhi University", "Harvard University")
           + "\nGender: Male\nAge: 52\nDate of Birth: 03/04/1972\nMarital Status: Married\n")
assert variant != orig
idx.add_text("web_02_renamed", variant)
res = {r["id"]: r for r in match_all(idx, ROLE, JD, top_k=50)["results"]}
a, b = res["web_02"], res["web_02_renamed"]
print(f"original score {a['score']}  | name/college/gender/age changed score {b['score']}")
assert a["score"] == b["score"] and a["matched"] == b["matched"] and a["breakdown"] == b["breakdown"], "BIAS!"
print("PASS bias: identical score, matches and breakdown")

# ---- 2. incremental --------------------------------------------------------------------------------------
before = {p.name: p.stat().st_mtime_ns for p in (tmp / "index").glob("*.pkl")}
new = """Name: Test Person
Headline: Backend Developer

SKILLS
Python, Flask, PostgreSQL, Docker

EXPERIENCE
Backend Developer, Acme (Jan 2021 - Present)
- Built a Flask REST API on PostgreSQL deployed with Docker on AWS.
"""
assert idx.add_text("new_resume", new) is True
assert idx.add_text("new_resume", new) is False, "unchanged resume should be skipped"
after = {p.name: p.stat().st_mtime_ns for p in (tmp / "index").glob("*.pkl")}
changed = [k for k in after if before.get(k) != after[k]]
print("files written when adding one resume:", changed)
assert changed == ["new_resume.pkl"]
ids = [r["id"] for r in match_all(idx, "Backend Developer", "Python Flask developer with PostgreSQL and Docker", 3)["results"]]
print("new resume is searchable immediately; top 3 for a Flask/PostgreSQL JD:", ids)
assert "new_resume" in ids
print("PASS incremental")
shutil.rmtree(tmp)
