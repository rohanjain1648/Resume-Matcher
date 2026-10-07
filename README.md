# Resume Matcher

Paste a role + job description, get the top 3 resumes with a score, matched/missing skills, a one-line reason and a
supporting line from the resume. 38 sample resumes are in `resumes/` (web, data, sales, HR, marketing, QA, design, 5 real hand-made ones).

## Run it

```bash
pip install -r requirements.txt
python cli.py index resumes/          # parse + embed the resumes once (about 1 min on CPU)
python cli.py match --role "MERN Stack Developer" --jd "Looking for a developer with 3+ years ... Cloud is a plus."
python app.py                         # simple web page at http://127.0.0.1:5000  (+ POST /api/match, POST /api/resumes)
python cli.py add path/to/new_resume.txt   # add ONE resume; nothing else is rebuilt
python evaluate.py                    # the 2-JD check (writes eval/results.md)
python checks/test_bias_and_incremental.py
```

First run downloads the embedding model `BAAI/bge-base-en-v1.5` (about 440 MB, needs roughly 1 GB free RAM). On a
low-RAM machine set `MATCHER_MODEL=BAAI/bge-small-en-v1.5` (130 MB) and re-run `cli.py index` (entries made with a
different model are re-embedded automatically). Needs internet only for that and for the optional
Wikipedia lookup below; both degrade gracefully (see "What does not work").

## How the matching works

1. **Parse a resume** (`matcher/resume_parser.py`). Sections are split by headings. The `Name:` line, the education
   section, college lines and age/gender/DOB/marital lines are **deleted before anything is embedded or scored**, so
   they cannot influence the result (tested: same resume, different name, college, gender and age gives the identical
   score, matches and breakdown). Extracted: skill terms (skills section plus code-looking tokens such as `Node.js`,
   `AWS`, `C++` found in the job history), job titles, evidence sentences, and years of experience (union of dated job
   ranges, so overlaps are not double counted). A skill listed in the skills section but never used in any job is
   flagged "listed only" and gets 55% credit.
2. **Parse the JD** (`matcher/jd_parser.py`). A requirement is a run of content words between stopwords/ad-filler
   ("web apps", "MERN stack", "REST APIs", "Cloud"). Sentences with "plus / nice to have / preferred" make their
   requirements weight 0.4; the role field counts 1.5; "3+ years" gives the minimum experience. No skill list is used.
3. **Match each requirement to each resume**, taking the strongest of:
   * **exact / alias**: normalised tokens (`Node.js`=`nodejs`, `APIs`=`API`, `web apps` ~ `web applications`) -> 1.0.
   * **named in the requirement's definition** (`matcher/knowledge.py`): the requirement is looked up once in Wikipedia
     and cached (`knowledge_cache.json`). Resume skills that the definition names count as related. This is how
     "MERN" -> MongoDB, Express, React, Node.js works without any hand-written pair, and it works for skills I
     never thought of. Each named skill adds 0.3 (combined as a soft-OR, cap 0.8), so one stray mention is not enough.
   * **embedding similarity**: skill-to-skill similarity is turned into a z-score against every skill in the index
     (so "unusually close for this requirement", not a global magic cosine), and a resume sentence unusually close to the
     requirement counts too. Related credit is capped (0.5-0.8) so exact matches always beat related ones.
   * **look-alike guard**: two different single-word names where one is a prefix of the other (`java` / `javascript`)
     get no similarity credit.
4. **Weight requirements by rarity** (IDF-style): a requirement nearly every resume meets (e.g. "web apps") counts
   for less than one few resumes meet.
5. **Score** = 55% skill coverage + 15% experience fit + 15% title fit + 15% overall semantic fit of the whole resume
   vs the JD. If years are below the stated minimum the whole score is scaled by 0.6-1.0. Every score comes with its
   breakdown, the matched/missing lists with the reason ("Express counts as MERN", "listed in skills, not shown in any
   job"), a why-sentence and the best supporting line. Near-identical resumes are flagged instead of silently ranked
   twice.
6. **Incremental**: each resume is stored as its own pickle in `index/`, keyed by a hash of its text. Adding one resume
   parses and embeds that file only (verified in `checks/`); corpus statistics are recomputed in memory in milliseconds.
   No training anywhere.

"Present" in a date range is resolved to 2025-01-01 (`MATCHER_AS_OF`, or `today`) so the sample results are
reproducible; the sample resumes were written around then.

## Resumes (`resumes/`)

38 text files: 12 web/mobile/devops, 8 data, 5 sales, 5 HR, 1 marketing, 1 QA, 1 design (`web_*`, `data_*`, `sales_*`,
`hr_*`, `mkt_01`, `qa_01`, `design_01`: AI-drafted, deliberately varied in style) plus **5 real, hand-made resumes**
(`manual_01`..`manual_05`: students / early-career developers; one is an SDE II). Experience ranges from 0 to 11 years.
The 5 real ones were supplied as PDFs (kept next to the `.txt` copies). `tools/pdf_to_txt.py` converts them: text
extraction, contact details (email, phone, links) removed, a `Name:` line added. One extra PDF
(`Anupam Ghosh_...pdf`, a course-completion letter, not a resume) was skipped by the same script. They have different
layouts (multi-column, titles on a separate line from the dates, "Languages: ..." sub-labels, projects instead of jobs),
which is what drove the parser improvements for headings, skill labels and multi-line job headers.
Tricky cases: `web_02` (Node, React, MongoDB, Express, never says "MERN"), `web_07` (about 38 skills, no experience),
`web_08` / `web_09` (near-identical), `web_05` (Java vs JavaScript), `data_04` / `hr_03` / `sales_03` (freshers),
`manual_03` / `manual_04` (MERN-style skills shown only in projects, no dated job).
`data_05`, `sales_03`, `hr_03` are free-form paragraphs with no headings.

## 2-JD check (`eval/expected.md` written first, results in `eval/results.md`)

Expected rankings (top 3 scored here; I also wrote down ranks 4-5) were written before the run (JD 1 is not fully blind: it is the brief's sample and I had run it while
debugging; JD 2 is blind). Baseline = count of JD words that appear verbatim in the resume.

| | Matcher | Keyword baseline |
|---|---|---|
| JD 1 MERN, overlap with my expected top 3 | 2/3 (`web_01`, `web_02` as expected; rank 3 is `web_06`, Angular+Node+Express, which I had at 5) | 2/3 (rank 3 is `web_03`) |
| JD 2 Data Analyst, overlap with expected top 3 | 3/3 (order as expected) | 2/3 (rank 3 is `data_04`, a 1-year fresher, for a 2+ years role) |

Numbers above are with `bge-base`. With the smaller `bge-small` the matcher scored 3/3 on JD 1 (the Vue pair at 3-4 and
`web_06` at 5) and 3/3 on JD 2: thresholds were tuned on `bge-small`, so the bigger model reshuffles the close calls
(`web_06` vs the `web_08/09` pair differ by about 6 points). I did not re-tune to match my expectations. `web_06` at 3 is
defensible: it shows web apps in its work history, which the Vue pair does not.

Where the matcher did better:
* JD 1: found `web_02` (Node/React/Mongo/Express, no "MERN") at rank 2 with a reason. A pure keyword search only
  gets it because it shares "web", "REST", "apps".
* JD 1: both Vue + Express + MongoDB resumes rank 3-4 as "MERN-like", flagged as near-identical, while the
  baseline lets a React-only, non-MongoDB resume in.
* Skill-stuffer (`web_07`) stays out of the top 3 on the 4 JDs I tried (rank 14 on MERN, 31 on Data Analyst); its
  skills are marked "listed only". On a JD that just lists many skills (React, Node, Mongo, AWS, Docker, Kubernetes,
  Python, Django) it still reaches rank 8, because its skills line really does contain all of them.
* Java JD: the Java engineer is first and JavaScript resumes get no credit for "Java".

Where it did worse / got something wrong:
* JD 2 (beyond the top 3): ranks `data_07` (statistician, R/Shiny dashboards) above `data_05` (data engineer, SQL) and puts
  `data_02` fifth; I expected `data_05` at 5. Dashboard words in `data_07`'s skills line pulled it up. The baseline also missed this
  one, so no gain there.
* JD 2: the baseline put the 1-year fresher `data_04` third (a real top-3 mistake); the matcher correctly pushes that person down (experience
  shortfall), but still shows it at rank 7 of 38, not lower.
* Noisy "related" labels: e.g. "Kubernetes (counts as Spring Boot)" and "AWS (counts as REST APIs)" in lower-ranked
  candidates. The related credit is small, so the ranking is not hurt, but the label is wrong and a recruiter would
  notice.
* `web_02` is shown "REST APIs listed in skills, not shown in any job" although a bullet says "REST endpoints":
  phrase matching is stricter than a human reader.

## What does not work yet

* Related-skill detection is only as good as (a) the Wikipedia definition of the requirement and (b) a small
  embedding model. If a skill has no Wikipedia page, or the lookup picks the wrong sense, quality falls back to the
  embedding score, which cannot reliably tell Java from Python (both "programming languages"). Offline, the MERN
  example would not rank `web_02` second.
* Lookups happen only for JD requirements, not for every resume skill; the `knowledge_cache.json` shipped here holds
  the lookups for the two evaluation JDs plus a couple of probes (it is a cache, safe to delete).
* Requirement extraction is "runs of content words": noisy on long, wordy JDs ("strong communication" is fine,
  "ability to work in fast paced environments" yields junk requirements). Years are read from dates; resumes with
  no dates and no stated years count as 0.
* "Experience" is total years, not years in the relevant skill.
* One language only (English), only plain `.txt` resumes (no PDF parsing), 38 resumes (no scale testing; scoring is a
  few matrix products per resume, fine up to thousands but not tuned).
* Weights (55/15/15/15, thresholds) were set by hand on these two JDs plus a few probes, so they are over-fitted
  to this small set; there is no labelled data to tune properly.

## What I would do next

1. Skill normalisation from a proper taxonomy (ESCO / O*NET) to replace the Wikipedia lookup, cached offline, and
   per-skill years of experience.
2. A cross-encoder re-ranker on the top 20 and a small labelled set (recruiter-judged pairs) to tune the weights.
3. PDF/DOCX parsing, and fairness tests across more fields (e.g. synthetic resumes differing only in protected
   attributes, and a check that location/phone/email are ignored too).
4. Cleaner explanations: suppress low-confidence "counts as" labels and show each requirement's own evidence line.
