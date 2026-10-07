"""Tiny web UI + JSON API.   python app.py   ->  http://127.0.0.1:5000

POST /api/match   {"role": "...", "jd": "...", "top": 5}   -> ranked candidates with scores and reasons
POST /api/resumes {"id": "new_resume", "text": "..."}       -> index ONE new resume (nothing else is rebuilt)
"""
import html

from flask import Flask, jsonify, request

from matcher.index import ResumeIndex
from matcher.scoring import match_all

app = Flask(__name__)
index = ResumeIndex("index")

PAGE = """<!doctype html><title>Resume Matcher</title>
<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}textarea,input{width:100%%;box-sizing:border-box}
.c{border:1px solid #ccc;border-radius:6px;padding:.7em 1em;margin:1em 0}small{color:#555}</style>
<h2>Resume Matcher</h2>
<form method=post action="/">
<label>Role<br><input name=role value="%(role)s"></label><br><br>
<label>Job description<br><textarea name=jd rows=7>%(jd)s</textarea></label><br><br>
<button>Find top 3</button> <small>%(n)d resumes indexed</small></form>%(out)s"""


def render(out):
    parts = [f"<p><small>Requirements: {html.escape('; '.join(r['text'] + (' (nice to have)' if r['weight'] < 1 else '') for r in out['requirements']))}</small></p>"]
    for r in out["results"]:
        b = "".join(f"<li>{html.escape(k)}: {html.escape(v)}</li>" for k, v in r["breakdown"].items())
        dup = f"<br><b>Note:</b> near-identical to {', '.join(r['near_duplicates'])}" if r.get("near_duplicates") else ""
        parts.append(
            f"<div class=c><b>Rank {r['rank']} · Score {r['score']:.0f}</b> · {html.escape(r['id'])} "
            f"<small>({html.escape(r['headline'])})</small><br>"
            f"<b>Matched:</b> {html.escape(', '.join(r['matched']) or '-')}<br>"
            f"<b>Missing:</b> {html.escape(', '.join(r['missing']) or '-')}<br>"
            f"<b>Why:</b> {html.escape(r['why'])}{dup}<br><b>Evidence:</b> &ldquo;{html.escape(r['evidence'])}&rdquo;"
            f"<details><summary>score breakdown</summary><ul>{b}</ul></details></div>")
    return "".join(parts)


@app.route("/", methods=["GET", "POST"])
def home():
    role, jd, out = request.form.get("role", ""), request.form.get("jd", ""), ""
    if request.method == "POST" and (role or jd):
        out = render(match_all(index, role, jd, 3))
    return PAGE % {"role": html.escape(role), "jd": html.escape(jd), "out": out, "n": len(index.docs)}


@app.post("/api/match")
def api_match():
    j = request.get_json(force=True)
    return jsonify(match_all(index, j.get("role", ""), j.get("jd", ""), int(j.get("top", 3))))


@app.post("/api/resumes")
def api_add():
    j = request.get_json(force=True)
    added = index.add_text(j["id"], j["text"])
    return jsonify({"added": added, "total": len(index.docs)})


if __name__ == "__main__":
    app.run(debug=False)
