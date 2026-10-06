"""Incremental resume index.

Each resume is parsed + embedded once and stored in its own pickle in `index/`, keyed by a hash of its text.
Adding a resume = parse and embed that one file. Nothing else is rebuilt or retrained.
The corpus-wide statistics (skill vocabulary matrix etc.) are cheap and recomputed in memory on load.
"""
import hashlib
import pathlib
import pickle
from datetime import date

import numpy as np

from . import embedder
from . import resume_parser, textutil
from .resume_parser import DEFAULT_AS_OF, parse_resume
from .textutil import phrase_key

# parser/text-helper changes invalidate cached parses automatically
_CODE = hashlib.sha1(b"".join(pathlib.Path(m.__file__).read_bytes() for m in (resume_parser, textutil))).hexdigest()[:10]


class ResumeIndex:
    def __init__(self, index_dir="index", as_of: date = None):
        self.dir = pathlib.Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.as_of = as_of or DEFAULT_AS_OF
        self.docs = {}
        self._load_all()

    # ------------------------------------------------------------------ storage
    def _load_all(self):
        self.docs = {}
        for p in sorted(self.dir.glob("*.pkl")):
            with open(p, "rb") as f:
                d = pickle.load(f)
            self.docs[d["id"]] = d
        self._build_vocab()

    def add_text(self, doc_id: str, text: str, source: str = "") -> bool:
        """Parse + embed one resume. Returns False if an identical resume is already indexed."""
        h = hashlib.sha1((embedder.MODEL_NAME + "|" + str(self.as_of) + "|" + _CODE + "|" + text).encode("utf8")).hexdigest()
        old = self.docs.get(doc_id)
        if old and old["hash"] == h:
            return False
        p = parse_resume(text, self.as_of)
        term_vecs = embedder.encode(p["terms"])
        bullet_vecs = embedder.encode(p["bullets"])
        title_vecs = embedder.encode(p["titles"])
        doc_text = (p["headline"] + ". " + p["skills_line"] + ". " + " ".join(p["bullets"]))[:2000]
        doc_vec = embedder.encode([doc_text])[0]
        d = {"id": doc_id, "hash": h, "source": source, "text": text, "parsed": p, "term_vecs": term_vecs,
             "bullet_vecs": bullet_vecs, "title_vecs": title_vecs, "doc_vec": doc_vec}
        with open(self.dir / f"{doc_id}.pkl", "wb") as f:
            pickle.dump(d, f)
        self.docs[doc_id] = d
        self._build_vocab()
        return True

    def add_file(self, path) -> bool:
        path = pathlib.Path(path)
        return self.add_text(path.stem, path.read_text(encoding="utf8"), str(path))

    def add_folder(self, folder) -> int:
        n = 0
        for p in sorted(pathlib.Path(folder).glob("*.txt")):
            n += self.add_file(p)
        return n

    # ------------------------------------------------------------------ corpus statistics
    def _build_vocab(self):
        keys, vecs, seen = [], [], set()
        for d in self.docs.values():
            for t, v in zip(d["parsed"]["terms"], d["term_vecs"]):
                k = phrase_key(t)
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
                    vecs.append(v)
        self.vocab_keys = keys
        self.vocab = np.vstack(vecs) if vecs else np.zeros((0, embedder.DIM), np.float32)
        allb = [d["bullet_vecs"] for d in self.docs.values() if len(d["bullet_vecs"])]
        self.bullet_bg = np.vstack(allb) if allb else np.zeros((0, embedder.DIM), np.float32)
        allt = [d["title_vecs"] for d in self.docs.values() if len(d["title_vecs"])]
        self.title_bg = np.vstack(allt) if allt else np.zeros((0, embedder.DIM), np.float32)
        self.doc_ids = list(self.docs)
        self.doc_bg = np.vstack([self.docs[i]["doc_vec"] for i in self.doc_ids]) if self.docs else np.zeros((0, embedder.DIM), np.float32)
