"""CloudSec Knowledge Graph Assistant engine (Module 10 In-Class Activity).

Loads the security knowledge base, builds the hybrid retrieval index once,
and answers questions as schema-validated JSON:
    {question, answer, route, graph_facts, sources, flagged_sources}

Same logic as the notebook: multi-hop count questions go to an exact graph
query; procedure questions go to hybrid (BM25 + TF-IDF) retrieval over the
runbooks; any retrieved runbook containing injected instructions is flagged
and never used as a source.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CONFIG = json.loads((HERE / "config.json").read_text())   # settings exported from the notebook

MAX_QUESTION_CHARS = 500          # reject absurdly long input
TIME_BUDGET_SECONDS = 5.0         # answers slower than this are reported as a timeout


# ---------------------------------------------------------------- JSON contract
class AskResponse(BaseModel):
    """The JSON contract every answer must satisfy."""
    question: str
    answer: str
    route: str = Field(pattern="^(graph|retrieval|none|error)$")
    graph_facts: list[str] = []
    sources: list[str] = []
    flagged_sources: list[str] = []
    latency_ms: float = 0.0


# ---------------------------------------------------------------- retrieval toolkit
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text):
    """Lowercase word tokens; IDs such as t1190 and cve-sim-2026-0001 stay intact."""
    return TOKEN.findall(text.lower())


def chunk_text(text, size=80, overlap=20):
    """Split a document into windows of `size` words that overlap by `overlap` words."""
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def build_chunks(docs, size=80, overlap=20):
    """One row per chunk; doc_id is kept so every answer can cite its source."""
    rows = [(d, j, c) for d, t in zip(docs.doc_id, docs.text) for j, c in enumerate(chunk_text(t, size, overlap))]
    return pd.DataFrame(rows, columns=["doc_id", "chunk_no", "text"])


class VectorIndex:
    """TF-IDF vectors + cosine similarity."""
    def __init__(self, chunks):
        self.chunks = chunks.reset_index(drop=True)
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2),
                                   sublinear_tf=True, token_pattern=None)
        self.M = self.vec.fit_transform(self.chunks.text)

    def scores(self, query):
        return linear_kernel(self.vec.transform([query]), self.M).ravel()


class BM25Index:
    """Okapi BM25 keyword scoring (k1 = 1.5, b = 0.75)."""
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks.reset_index(drop=True)
        self.cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.cv.fit_transform(self.chunks.text).astype(float).tocsr()
        dl = np.asarray(tf.sum(axis=1)).ravel()
        n, df = tf.shape[0], np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[rows] / dl.mean()))
        self.W = tf.tocsc()

    def scores(self, query):
        ids = [self.cv.vocabulary_[w] for w in tokenize(query) if w in self.cv.vocabulary_]
        if not ids:
            return np.zeros(self.W.shape[0])
        return np.asarray(self.W[:, ids] @ self.idf[ids]).ravel()


def minmax(x):
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


# ---------------------------------------------------------------- grounding helpers
GRAPH_Q = re.compile(r"internet-facing (\w+) assets have an unpatched (\w+) vulnerability that enables .*\((T\d+)\)",
                     flags=re.I)
INJECTION_PATTERNS = CONFIG["injection_filter"]
FIELD_Q = re.compile(r"\bwho\b|how long|deadline|time limit|minutes|escalat|\bpage", flags=re.I)  # fields runbooks hold


def is_injected(text):
    """Retrieved text is data, never instructions: flag anything that tries to command the assistant."""
    return any(re.search(p, text, flags=re.I) for p in INJECTION_PATTERNS)


def extract_answer(question, doc_text):
    """Extractive reader: copy the requested field (escalation contact or time limit) from a runbook."""
    if re.search(r"\bwho\b", question, flags=re.I):
        m = re.search(r"page ([a-z]+-lead)", doc_text)
        return m.group(1) if m else None
    m = re.search(r"within (\d+) minutes", doc_text)
    return f"{m.group(1)} minutes" if m else None


# ---------------------------------------------------------------- engine
class Engine:
    """Loads the knowledge base and builds the indexes once; ask() answers one question."""

    def __init__(self, data_dir=DATA_DIR):
        assets = pd.read_csv(data_dir / "assets.csv")
        cves = pd.read_csv(data_dir / "cves.csv")
        asset_vulns = pd.read_csv(data_dir / "asset_vulns.csv")
        techniques = pd.read_csv(data_dir / "techniques.csv")
        self.runbooks = pd.read_csv(data_dir / "runbooks.csv")
        self.qa = pd.read_csv(data_dir / "security_qa_300.csv")
        # Graph view: every UNPATCHED asset -> CVE -> technique link with asset details
        self.open_vulns = (asset_vulns[~asset_vulns.patched]
                           .merge(cves, on="cve_id").merge(assets, on="asset_id")
                           .merge(techniques[["technique_id", "technique_name"]], on="technique_id"))
        chunks = build_chunks(self.runbooks, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
        self.vindex, self.bindex = VectorIndex(chunks), BM25Index(chunks)
        self.doc_text = dict(zip(self.runbooks.doc_id, self.runbooks.text))
        self.n_docs = len(self.runbooks)

    def ranked_docs(self, query, k=3, alpha=0.5):
        """Hybrid score (BM25 + TF-IDF), then the first k distinct runbooks."""
        s = alpha * minmax(self.vindex.scores(query)) + (1 - alpha) * minmax(self.bindex.scores(query))
        out, seen = [], set()
        for i in np.argsort(-s):
            d = self.vindex.chunks.doc_id.iat[i]
            if d not in seen:
                seen.add(d)
                out.append((d, float(s[i])))
            if len(out) == k:
                break
        return out

    def graph_answer(self, question):
        env, sev, tid = GRAPH_Q.search(question).groups()
        ov = self.open_vulns
        hits = ov[(ov.environment == env.lower()) & (ov.severity.str.lower() == sev.lower())
                  & (ov.technique_id == tid.upper()) & ov.internet_facing]
        return str(hits.asset_id.nunique()), hits

    def _answer(self, question, k):
        if GRAPH_Q.search(question):                                   # multi-hop count -> exact graph query
            ans, hits = self.graph_answer(question)
            facts = [f"{r.asset_id} --hasOpenVuln--> {r.cve_id} --enables--> {r.technique_id}"
                     for r in hits.head(5).itertuples()]
            return dict(answer=ans, route="graph", graph_facts=facts, sources=["asset-graph"])
        if not FIELD_Q.search(question):                               # out of scope -> do not guess
            return dict(answer="I can answer exposure counts, runbook time limits and escalation contacts. "
                               "Try one of the example questions.", route="none")
        if self.bindex.scores(question).max() == 0 and self.vindex.scores(question).max() == 0:
            return dict(answer="Not found in the runbooks.", route="none")   # nothing relevant retrieved
        flagged, trusted = [], []
        for d, _ in self.ranked_docs(question, k):                     # retrieve candidate runbooks
            (flagged if is_injected(self.doc_text[d]) else trusted).append(d)
        for d in trusted:                                              # first trusted source with the field wins
            a = extract_answer(question, self.doc_text[d])
            if a:
                return dict(answer=a, route="retrieval", sources=[d], flagged_sources=flagged)
        return dict(answer="Not found in the runbooks.", route="none", flagged_sources=flagged)

    def ask(self, question, k=None):
        """Validated JSON for any input: bad input, timeouts and internal errors return route='error'."""
        k = k or CONFIG["top_k"]
        q = (question or "").strip()
        t0 = time.perf_counter()
        if not q:
            return AskResponse(question="", answer="Please enter a question.", route="error")
        if len(q) > MAX_QUESTION_CHARS:
            return AskResponse(question=q[:80] + "...", route="error",
                               answer=f"Question is too long (limit {MAX_QUESTION_CHARS} characters).")
        try:
            result = self._answer(q, k)
        except Exception as exc:                                       # never crash the app on one question
            return AskResponse(question=q, answer=f"Internal error: {type(exc).__name__}", route="error")
        ms = 1000 * (time.perf_counter() - t0)
        if ms > TIME_BUDGET_SECONDS * 1000:
            return AskResponse(question=q, route="error", latency_ms=round(ms, 1),
                               answer=f"Timed out (over {TIME_BUDGET_SECONDS:.0f} s). Try a shorter question.")
        return AskResponse(question=q, latency_ms=round(ms, 1), **result)
