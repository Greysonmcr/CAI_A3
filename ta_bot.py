"""
CS 6604 / ECE 6256 — Assignment 3: The Course TA-Bot
Conversational RAG with Memory

Pipeline per turn
-----------------
1. Memory write   — extract durable, non-sensitive facts about the user from the
                    message (rule-based, plus optional LLM extraction that must
                    quote the user verbatim). Sensitive content is refused.
2. Intent         — memory-recall questions ("do you remember ...") are answered
                    from the memory store without retrieval.
3. Query building — standalone query from the message; in multi-turn sessions an
                    LLM rewrite (and a heuristic history expansion) is added when
                    the message depends on earlier turns.
4. Retrieval      — hybrid paper ranking fused with weighted reciprocal-rank
                    fusion over: chunk BM25 (corpus FTS5), paper BM25 over
                    title+abstract, and a name/alias matcher (e.g. "ColBERT",
                    "DPR"). Optional dense retrieval through /v1/embeddings.
5. Generation     — answer grounded in numbered excerpts; citations are the
                    retrieved paper_ids.

Do NOT rename the class or change the method signatures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

from retrieval import Retriever

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
STOPWORDS = set(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just let me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own please same she should so some
    such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom
    why will with would you your yours yourself yourselves explain describe tell know
    paper papers discuss discussed course lecture according approach work works use used
    using way ways does make makes like give compare comparison mean means summarize
    summary key main different difference differences actually going back switch topics
    topic regarding related relate relates question questions answer can could us thing
    things say says said show shows find found get
    """.split()
)

ANAPHORA = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|both|either|neither|the former|"
    r"the latter|above|earlier|previous|previously|aforementioned|same|such|we discussed|"
    r"you mentioned|you said|so far|all three|all two|all four|the two|the three|each of)\b",
    re.I,
)
BACKREF = re.compile(
    r"\b(we discussed|we('ve| have) discussed|you mentioned|so far|all (two|three|four|five|of them)|"
    r"(the|these|those) (two|three|four|five)|both (of )?(these|those|them|approaches|methods|papers)|"
    r"each of (these|those|them)|compare them|between them)\b",
    re.I,
)
TOPIC_SWITCH = re.compile(
    r"\b(switch(ing)? topics?|change (the )?(subject|topic)|different (topic|question)|new topic|"
    r"unrelated question|something else)\b",
    re.I,
)
NUMBER_WORDS = {"two": 2, "both": 2, "three": 3, "four": 4, "five": 5}

GENERIC_ALIASES = {
    "LLM", "LLMS", "LM", "LMS", "NLP", "AI", "QA", "ML", "RL", "NMT", "RNN", "RNNS",
    "IR", "NLU", "SGD", "GPT", "NN", "DL", "CNN", "LSTM",
}
# Mixed-case aliases that are also ordinary English words: match case-sensitively only.
WORDLIKE_ALIASES = {"react", "bagel", "lima", "tofu", "medusa", "moshi", "ragas", "usr"}


def content_terms(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def fts_query(text: str) -> str | None:
    terms = list(dict.fromkeys(content_terms(text)))
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms[:40])


# ---------------------------------------------------------------------------
# Memory: extraction rules and refusal filter
# ---------------------------------------------------------------------------
_END = r"(?=(?:[.!?;]\s|[.!?;]?$|,?\s+(?:and|but)\s+(?:i|i'm|i am|my|me)\b|,\s*(?:so|which)\b))"

MEMORY_RULES: list[tuple[str, str, re.Pattern]] = [
    # (slot, label, pattern) — group "v" is the stored value, copied verbatim
    ("name", "Name",
     re.compile(r"\b(?i:my name is|my name's|call me|i go by)\s+(?P<v>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})")),
    ("name", "Name",
     re.compile(r"(?:^|[.!?]\s+)(?i:hi|hello|hey)?[, ]*(?i:i'm|i am)\s+(?P<v>[A-Z][a-z'-]+)(?:[,.!]|\s+and\b|$)")),
    ("project", "Project",
     re.compile(r"\b(?:my|our)\s+(?:semester |final |course |class |research |term |capstone )?"
                r"(?:project|thesis|paper|dissertation|research)\s+(?:is|will be)\s+(?:on|about|focused on|related to)?\s*"
                r"(?P<v>.+?)" + _END, re.I)),
    ("project", "Project",
     re.compile(r"\b(?:i'm|i am|we're|we are)\s+(?:writing|doing|working on|building)\s+(?:my|our|a)\s+"
                r"(?:semester |final |course |class |research |term |capstone )?(?:project|thesis|paper|report)\s+"
                r"(?:on|about)\s+(?P<v>.+?)" + _END, re.I)),
    ("interest", "Interested in",
     re.compile(r"\b(?:i'm|i am)\s+(?:(?:most|really|very|mostly|particularly|especially|mainly|primarily)\s+)?"
                r"(?:interested in|curious about|into|passionate about|focused on|focusing on)\s+(?P<v>.+?)" + _END, re.I)),
    ("interest", "Interested in",
     re.compile(r"\bi\s+(?:really\s+)?(?:like|love|enjoy|care about)\s+(?P<v>.+?)" + _END, re.I)),
    ("interest", "Interested in",
     re.compile(r"\bmy\s+(?:main\s+|primary\s+|research\s+)?(?:interest|interests|focus|area)\s+(?:is|are|lies in)\s+"
                r"(?P<v>.+?)" + _END, re.I)),
    ("background", "Background",
     re.compile(r"\b(?:i'm|i am)\s+(?P<v>an?\s+(?:[\w'-]+\s+){0,3}(?:student|undergrad|undergraduate|phd|master's|masters|"
                r"researcher|engineer|developer|scientist|ta|instructor)\b(?:\s+(?:in|at|studying|working on)\s+[^.!?;,]+)?)", re.I)),
    ("background", "Background",
     re.compile(r"\bmy (?:background|major) is (?:in )?(?P<v>.+?)" + _END, re.I)),
    ("background", "Background",
     re.compile(r"\bi (?:work as|work on|major in|study)\s+(?P<v>.+?)" + _END, re.I)),
    ("goal", "Goal",
     re.compile(r"\b(?:i'm|i am)\s+(?:preparing for|studying for|trying to|planning to|hoping to)\s+(?P<v>.+?)" + _END, re.I)),
    ("goal", "Goal",
     re.compile(r"\bmy goal is (?:to )?(?P<v>.+?)" + _END, re.I)),
    ("preference", "Prefers",
     re.compile(r"\bi (?:prefer|would prefer|'d prefer|like it when you)\s+(?P<v>.+?)" + _END, re.I)),
    ("preference", "Prefers",
     re.compile(r"\bplease (?:always\s+)?(?P<v>(?:keep|give|use|explain|answer|include|avoid)\b.+?)" + _END, re.I)),
]
SINGLE_VALUED = {"name", "project"}

# What we refuse to store: credentials, identifiers, contact data, special-category
# personal data (health, religion, sexuality, politics, immigration, finances,
# grades/disciplinary records), and third-party personal information.
SENSITIVE_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",                                 # SSN
    r"\b(?:\d[ -]?){9,19}\b",                                 # long digit runs (cards, IDs, phones)
    r"\+?\d[\d ().-]{7,}\d",                                  # phone numbers
    r"[\w.+-]+@[\w-]+\.[\w.]+",                               # email
    r"\b(?:password|passcode|passwd|pin|api[ _-]?key|secret|token|credentials?|login)\b",
    r"\b(?:ssn|social security|student id|gt ?id|gtid|passport|driver'?s licen[cs]e|bank|routing|"
    r"account number|credit card|debit card|iban)\b",
    r"\b(?:address is|i live (?:at|on)|my address|home address|zip code|apartment)\b",
    r"\b(?:date of birth|birthday|born on|dob)\b",
    r"\b(?:diagnos\w*|disorder|disease|illness|depress\w*|anxiety|adhd|autis\w*|dyslexi\w*|"
    r"medication|therapy|therapist|pregnan\w*|disabilit\w*|mental health|chronic|cancer|hiv|"
    r"medical|surgery|hospital\w*|prescri\w*)\b",
    r"\b(?:religio\w*|christian|muslim|jewish|hindu|buddhist|atheist|church|mosque|synagogue)\b",
    r"\b(?:gay|lesbian|bisexual|transgender|queer|sexual orientation|sexuality)\b",
    r"\b(?:democrat\w*|republican\w*|political party|vote[ds]? for)\b",
    r"\b(?:visa|immigration|undocumented|green card|citizenship status)\b",
    r"\b(?:salary|income|debt|loan|financial aid|bankrupt\w*)\b",
    r"\b(?:gpa|my grades?|failing|academic probation|honor code violation|plagiari\w*|cheat\w*)\b",
    r"\b(?:my (?:girlfriend|boyfriend|wife|husband|partner|roommate|friend|mom|dad|mother|father|"
    r"brother|sister|ex)\b)",
]
SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.I)

NO_STORE_RE = re.compile(
    r"\b(?:don'?t|do not|never|please don'?t)\s+(?:remember|store|save|record|keep|memori[sz]e|note)\b|"
    r"\boff the record\b|\b(?:just|only) for (?:this|now|today)\b|\bhypothetically\b|\bpretend\b|"
    r"\bimagine (?:that )?i\b|\bjust kidding\b|\bnot really\b",
    re.I,
)
FORGET_RE = re.compile(
    r"\b(?:forget|delete|erase|remove|wipe|clear)\b\s+(?P<what>.*)", re.I
)
FORGET_ALL_RE = re.compile(
    r"\b(?:forget|delete|erase|wipe|clear)\s+(?:everything|all)\b|\b(?:forget|delete|erase|wipe|clear)\s+"
    r"(?:all\s+)?(?:of\s+)?(?:my|your)\s+(?:memory|memories|data|information|info)\b",
    re.I,
)
RECALL_RE = re.compile(
    r"\b(?:do you |can you |what do you |what can you )?(?:remember|recall)\b|"
    r"\bwhat do you know about me\b|\bwho am i\b|\bremind me\b|\bwhat(?:'s| is| are| was) my (?:name|project|thesis|"
    r"background|goals?|interests?|focus|major|preferences?)\b|"
    r"\bwhat (?:am i|was i|i'm|i am|i was) (?:working on|interested in|writing|studying|into)\b|\babout me\b",
    re.I,
)


class _EndpointDown(Exception):
    pass


# ---------------------------------------------------------------------------
# TABot
# ---------------------------------------------------------------------------
class TABot:
    """Course TA-Bot with RAG retrieval and cross-session memory."""

    def __init__(self, config: dict) -> None:
        """
        Required config keys (set by the harness): endpoint, model, corpus_path,
        temperature, seed.

        Optional keys (defaults in brackets):
            timeout          [60]     seconds per HTTP request
            max_citations    [3]      max papers cited per answer
            cite_ratio       [0.6]    cite paper i if fused score >= ratio * top score
            rankers          [["chunk", "paper", "alias"]]  (+ "dense")
            chunk_retriever  [{"remove_stopwords": True, "and_first": False}]  Retriever options
            ranker_weights   [{"chunk": 1.0, "paper": 1.0, "alias": 1.5, "dense": 1.0, "history": 0.5}]
            rewrite          ["auto"] "auto" | "always" | "off" — LLM query rewriting
            llm_memory       [True]   LLM-assisted (verbatim-quoted) fact extraction
            min_quality      [0.0]    drop chunks with quality below this from the context
            embedding_model  [model]  model name for /v1/embeddings (dense ranker)
            cache_dir        [None]   where to cache chunk embeddings
            memory_path      [None]   JSON file to persist memory across processes
            max_tokens       [600]
        """
        self.config = config
        ep = str(config["endpoint"]).rstrip("/")
        self.base_url = ep if ep.endswith("/v1") else ep + "/v1"
        self.model = config["model"]
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 0))
        self.timeout = float(config.get("timeout", 60))
        self.max_tokens = int(config.get("max_tokens", 600))
        self.max_citations = int(config.get("max_citations", 3))
        self.cite_ratio = float(config.get("cite_ratio", 0.6))
        self.rankers = list(config.get("rankers", ["chunk", "paper", "alias"]))
        self.weights = {"chunk": 1.0, "paper": 1.0, "alias": 1.5, "dense": 1.0, "history": 0.5}
        self.weights.update(config.get("ranker_weights", {}))
        self.rewrite_mode = config.get("rewrite", "auto")
        self.llm_memory = bool(config.get("llm_memory", True))
        self.min_quality = float(config.get("min_quality", 0.0))
        self.memory_path = config.get("memory_path")
        # Query options for the chunk retriever (see retrieval.Retriever.retrieve)
        self.chunk_opts = {"remove_stopwords": True, "and_first": False}
        self.chunk_opts.update(config.get("chunk_retriever", {}))

        self._http = requests.Session()
        self._lock = threading.Lock()
        self._sessions: dict[str, list[dict]] = defaultdict(list)
        self._memory: dict[str, list[dict]] = {}
        self._load_memory()

        self._load_corpus(Path(config["corpus_path"]))
        self._dense = None
        if "dense" in self.rankers:
            self._dense = self._build_dense_index(config.get("cache_dir"))

    # ------------------------------------------------------------------
    # Corpus / indexes
    # ------------------------------------------------------------------
    def _load_corpus(self, corpus_path: Path) -> None:
        db = corpus_path / "corpus.db" if corpus_path.is_dir() else corpus_path
        self.db = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.retriever = Retriever(db)
        self.papers: dict[str, dict] = {}
        for pid, title, abstract, year, arxiv in self.db.execute(
            "SELECT paper_id, title, abstract, year, arxiv_id FROM papers ORDER BY paper_id"
        ):
            self.papers[pid] = {"title": title or "", "abstract": abstract or "", "year": year, "arxiv": arxiv}

        # Paper-level BM25 index (title weighted above abstract), in memory.
        self.pdb = sqlite3.connect(":memory:", check_same_thread=False)
        self.pdb.execute(
            "CREATE VIRTUAL TABLE pfts USING fts5(paper_id UNINDEXED, title, abstract, tokenize='porter unicode61')"
        )
        self.pdb.executemany(
            "INSERT INTO pfts VALUES (?,?,?)",
            [(pid, p["title"], p["abstract"]) for pid, p in self.papers.items()],
        )
        self._build_aliases()

    def _build_aliases(self) -> None:
        """Names a student would use to refer to a paper, derived from its title/abstract."""
        named = re.compile(
            r"\b(?:[Ww]e (?:propose|introduce|present|build|release|call)|called|named|dubbed|termed)\s+"
            r"(?:(?:an?|the|our)\s+)?(?:\w+\s+){0,2}?([A-Za-z]*[A-Z][a-z0-9]*[A-Z0-9][\w-]*)"
        )
        aliases: dict[str, set[str]] = defaultdict(set)
        for pid, p in self.papers.items():
            title = p["title"]
            cands = set()
            if ":" in title:
                pre = title.split(":")[0].strip()
                if 1 <= len(pre.split()) <= 4 and len(pre) >= 3 and "$" not in pre:
                    cands.add(pre)
            for m in named.finditer(p["abstract"]):
                cands.add(m.group(1))
            # Initials of the leading capitalised run, e.g. "Dense Passage Retrieval" -> DPR
            head = re.split(r"\s+(?:for|with|via|of|on|in|from|using|to|and|is|are|as)\s+", title.split(":")[-1].strip())[0]
            words = re.split(r"[\s-]+", head)
            if 2 <= len(words) <= 5 and all(w[:1].isupper() for w in words if w):
                ini = "".join(w[0] for w in words if w).upper()
                if len(ini) >= 3:
                    cands.add(ini)
            for c in cands:
                c = c.strip()
                if c.upper() in GENERIC_ALIASES or len(c) < 3:
                    continue
                if c.isupper() and len(c) <= 3 and c not in title:
                    continue
                aliases[c].add(pid)
        # Keep only aliases that identify a single paper.
        self.aliases: list[tuple[re.Pattern, str]] = []
        for alias, pids in aliases.items():
            if len(pids) != 1:
                continue
            flags = 0
            if (" " in alias or any(ch.isdigit() for ch in alias) or "-" in alias
                    or sum(ch.isupper() for ch in alias) >= 2) and alias.lower() not in WORDLIKE_ALIASES:
                flags = re.I
            pat = re.compile(r"(?<![\w-])" + re.escape(alias) + r"(?:'s)?(?![\w-])", flags)
            self.aliases.append((pat, next(iter(pids))))

    def _build_dense_index(self, cache_dir: str | None):
        rows = self.db.execute("SELECT chunk_id, paper_id, text FROM chunks ORDER BY chunk_id").fetchall()
        emb_model = self.config.get("embedding_model", self.model)
        key = hashlib.sha256(f"{emb_model}|{len(rows)}".encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"emb_{key}.npy" if cache_dir else None
        if cache and cache.exists():
            mat = np.load(cache)
        else:
            vecs = []
            for i in range(0, len(rows), 64):
                vecs.extend(self._embed([r[2][:2000] for r in rows[i:i + 64]]))
            mat = np.asarray(vecs, dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, mat)
        return {"pids": [r[1] for r in rows], "mat": mat}

    # ------------------------------------------------------------------
    # Rankers — each returns an ordered list of paper_ids (best first)
    # ------------------------------------------------------------------
    def _rank_chunk(self, query: str) -> list[str]:
        # Chunk-level BM25 via retrieval.Retriever; diversify=True keeps each paper's best chunk.
        try:
            rows = self.retriever.retrieve(query, k=30, diversify=True, **self.chunk_opts)
        except sqlite3.OperationalError:
            return []
        return [r["paper_id"] for r in rows]

    def _rank_paper(self, query: str) -> list[str]:
        q = fts_query(query)
        if not q:
            return []
        try:
            rows = self.pdb.execute(
                "SELECT paper_id FROM pfts WHERE pfts MATCH ? ORDER BY bm25(pfts, 0.0, 4.0, 1.0), paper_id LIMIT 30",
                (q,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]

    def _rank_alias(self, query: str) -> list[str]:
        hits = []
        for pat, pid in self.aliases:
            m = pat.search(query)
            if m:
                hits.append((m.start(), -len(m.group(0)), pid))
        out = []
        for _, _, pid in sorted(hits):
            if pid not in out:
                out.append(pid)
        return out

    def _rank_dense(self, query: str) -> list[str]:
        if self._dense is None:
            return []
        v = np.asarray(self._embed([query])[0], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        sims = self._dense["mat"] @ v
        out: list[str] = []
        for i in np.argsort(-sims, kind="stable")[:100]:
            pid = self._dense["pids"][i]
            if pid not in out:
                out.append(pid)
        return out

    def retrieve(self, query: str, history_query: str | None = None, rewritten: str | None = None) -> list[tuple[str, float]]:
        """Weighted reciprocal-rank fusion over the configured rankers."""
        lists: list[tuple[float, list[str]]] = []
        for q, scale in ((query, 1.0), (rewritten, 1.0)):
            if not q:
                continue
            for name in self.rankers:
                ranked = getattr(self, f"_rank_{name}")(q)
                lists.append((self.weights[name] * scale, ranked))
        if history_query:
            lists.append((self.weights["history"], self._rank_chunk(history_query)))
        fused: dict[str, float] = defaultdict(float)
        for w, ranked in lists:
            for r, pid in enumerate(ranked[:30]):
                fused[pid] += w / (10 + r)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

    # ------------------------------------------------------------------
    # Endpoint
    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self._http.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise _EndpointDown(str(exc)) from exc
        if r.status_code >= 500 or r.status_code in (408, 429):
            raise _EndpointDown(f"HTTP {r.status_code}")
        r.raise_for_status()
        return r.json()

    def _complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
        data = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "seed": self.seed,
                "max_tokens": max_tokens or self.max_tokens,
            },
        )
        msg = data["choices"][0]["message"]
        return (msg.get("content") or "").strip()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": self.config.get("embedding_model", self.model), "input": texts})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def _load_memory(self) -> None:
        if self.memory_path and os.path.exists(self.memory_path):
            try:
                with open(self.memory_path) as f:
                    self._memory = json.load(f)
            except (OSError, ValueError):
                self._memory = {}

    def _save_memory(self) -> None:
        if not self.memory_path:
            return
        tmp = f"{self.memory_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self._memory, f, indent=1)
        os.replace(tmp, self.memory_path)

    @staticmethod
    def _clean_value(v: str) -> str:
        return re.sub(r"\s+", " ", v).strip(" ,.;:!?\"'")[:200]

    def _rule_facts(self, message: str) -> list[tuple[str, str, str]]:
        facts = []
        for slot, label, pat in MEMORY_RULES:
            for m in pat.finditer(message):
                v = self._clean_value(m.group("v"))
                if len(v) >= 2 and not v.endswith("?"):
                    facts.append((slot, label, v))
        return facts

    def _llm_facts(self, message: str) -> list[tuple[str, str, str]]:
        prompt = (
            "Extract durable facts the student states about THEMSELVES that would help a course TA "
            "personalise future help: name, project/thesis topic, research interests, academic "
            "background, learning goals, answer-style preferences.\n"
            "Do NOT extract: questions, temporary states or moods, anything about other people, "
            "contact details, IDs, passwords, health, religion, politics, sexuality, finances, grades, "
            "or anything the student asks you not to remember.\n"
            'Return a JSON list like [{"slot": "project", "quote": "<exact words copied from the message>"}]. '
            "slot is one of name, project, interest, background, goal, preference. "
            "Return [] if there is nothing.\n\nMessage: " + json.dumps(message)
        )
        try:
            out = self._complete([{"role": "user", "content": prompt}], max_tokens=200, temperature=0.0)
        except _EndpointDown:
            raise
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return []
        m = re.search(r"\[.*\]", out, re.S)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except ValueError:
            return []
        labels = {s: l for s, l, _ in MEMORY_RULES}
        facts = []
        low = message.lower()
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            slot, quote = str(it.get("slot", "")).lower(), self._clean_value(str(it.get("quote", "")))
            # Grounding check: the quote must literally appear in what the user said.
            if slot in labels and len(quote) >= 2 and quote.lower() in low:
                facts.append((slot, labels[slot], quote))
        return facts

    def _update_memory(self, user_id: str, message: str) -> None:
        store = self._memory.setdefault(user_id, [])
        if FORGET_ALL_RE.search(message):
            store.clear()
            self._save_memory()
            return
        fm = FORGET_RE.search(message)
        if fm and re.search(r"\b(?:my|that|what i|about me|it)\b", fm.group("what"), re.I):
            what = set(content_terms(fm.group("what"))) - {"name", "remember", "told", "said", "mentioned"}
            slot_words = {"name": "name", "project": "project", "thesis": "project", "interest": "interest",
                          "interests": "interest", "background": "background", "goal": "goal", "preference": "preference"}
            slots = {slot_words[w] for w in content_terms(fm.group("what")) if w in slot_words}
            store[:] = [
                f for f in store
                if f["slot"] not in slots and not (what & set(content_terms(f["value"])))
            ]
            self._save_memory()
            return
        if NO_STORE_RE.search(message):
            return
        candidates = self._rule_facts(message)
        down = None
        if self.llm_memory and re.search(r"\b(i|i'm|i am|my|me|call me)\b", message, re.I):
            covered = {s for s, _, _ in candidates}
            try:
                candidates += [f for f in self._llm_facts(message) if f[0] not in covered]
            except _EndpointDown as exc:
                down = exc  # still keep the rule-based facts, then fail the turn
        changed = False
        for slot, label, value in candidates:
            # Refuse sensitive values; also refuse when the clause around it is sensitive.
            if SENSITIVE_RE.search(value) or SENSITIVE_RE.search(f"{label} {value}"):
                continue
            if any(f["value"].lower() == value.lower() for f in store):
                continue
            if any(value.lower() in f["value"].lower() for f in store if f["slot"] == slot):
                continue
            if slot in SINGLE_VALUED:
                store[:] = [f for f in store if f["slot"] != slot]
            else:
                store[:] = [f for f in store if not (f["slot"] == slot and f["value"].lower() in value.lower())]
            store.append({"slot": slot, "label": label, "value": value})
            changed = True
        if changed:
            del store[:-20]  # bounded store
            self._save_memory()
        if down:
            raise down

    @staticmethod
    def _fact_str(f: dict) -> str:
        return f"{f['label']}: {f['value']}"

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------
    def _needs_context(self, message: str, history: list[dict]) -> bool:
        if not history or TOPIC_SWITCH.search(message):
            return False
        return bool(ANAPHORA.search(message)) or len(content_terms(message)) <= 3

    def _rewrite(self, message: str, history: list[dict]) -> str | None:
        convo = "\n".join(f"User: {t['user']}" for t in history[-3:])
        prompt = (
            "Rewrite the user's latest message as a standalone search query for a corpus of NLP / "
            "conversational-AI research papers. Resolve pronouns and references like 'that approach' "
            "using the conversation, and name the specific methods or papers being referred to. "
            "If the latest message changes topic, do not carry over the old topic. "
            "Output only the query, on one line.\n\n"
            f"Conversation so far:\n{convo}\n\nLatest message: {message}\n\nStandalone query:"
        )
        try:
            out = self._complete([{"role": "user", "content": prompt}], max_tokens=60, temperature=0.0)
        except _EndpointDown:
            raise
        except (requests.RequestException, KeyError, IndexError, ValueError):
            return None
        out = out.strip().splitlines()[0].strip(" \"'") if out.strip() else ""
        # Sanity check: a usable rewrite is short and overlaps the conversation's vocabulary.
        known = set(content_terms(message + " " + convo))
        if not out or len(out.split()) > 40 or not (set(content_terms(out)) & known):
            return None
        return out

    # ------------------------------------------------------------------
    # Context + generation
    # ------------------------------------------------------------------
    def _best_chunks(self, query: str, pid: str, k: int = 2) -> list[tuple[str, str]]:
        q = fts_query(query)
        rows = []
        if q:
            try:
                rows = self.db.execute(
                    "SELECT f.section, c.text, c.quality FROM chunks_fts f JOIN chunks c USING(chunk_id) "
                    "WHERE chunks_fts MATCH ? AND f.paper_id = ? ORDER BY bm25(chunks_fts) LIMIT 8",
                    (q, pid),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        good = [(s, t) for s, t, qual in rows if (qual or 0) >= self.min_quality]
        return good[:k]

    def _build_context(self, query: str, pids: list[str]) -> str:
        blocks = []
        for i, pid in enumerate(pids, 1):
            p = self.papers[pid]
            parts = [f"[{i}] {p['title']} ({p['year']}; paper_id {pid})", f"Abstract: {p['abstract'][:1200]}"]
            for sec, text in self._best_chunks(query, pid):
                parts.append(f"Excerpt ({sec}): {text[:1800]}")
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    def _system_prompt(self, facts: list[str]) -> str:
        sp = (
            "You are the TA-Bot for a graduate course on conversational AI and large language models. "
            "Answer the student's question using the numbered course papers provided in the context. "
            "Rules:\n"
            "- Ground every claim in the context; cite papers inline with their number, e.g. [1] or [2].\n"
            "- Be specific: name the method, how it works, and concrete findings or numbers when the "
            "context gives them. Answer every part of the question.\n"
            "- If the context does not contain the answer, say so briefly instead of guessing.\n"
            "- Use the conversation history to resolve references like 'that approach'.\n"
            "- Keep it focused: about 120-250 words, plain prose or a short list."
        )
        if facts:
            sp += (
                "\n\nWhat you know about this student (from earlier sessions; use it only to tailor "
                "the answer when relevant, do not recite it unprompted):\n- " + "\n- ".join(facts)
            )
        return sp

    def _answer_recall(self, message: str, facts: list[str]) -> str:
        if not facts:
            return ("I don't have anything stored about you yet. If you tell me things like your name, "
                    "project topic, or interests, I can remember them for future sessions.")
        intro = ""
        try:
            intro = self._complete(
                [
                    {"role": "system", "content": "You are a friendly course TA-Bot. Answer the student's question "
                     "about what you remember, using ONLY these stored facts (quote them verbatim):\n- "
                     + "\n- ".join(facts)},
                    {"role": "user", "content": message},
                ],
                max_tokens=200,
            )
        except _EndpointDown:
            raise
        except (requests.RequestException, KeyError, IndexError, ValueError):
            intro = ""
        # Guarantee the stored facts appear verbatim even if the model paraphrases.
        missing = [f for f in facts if f.split(": ", 1)[-1].lower() not in intro.lower()]
        if missing or not intro:
            intro = (intro + "\n\n" if intro else "") + "Here's what I remember about you:\n- " + "\n- ".join(missing or facts)
        return intro

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, session_id: str, user_id: str, message: str) -> dict:
        """Returns {"answer": str, "citations": [paper_id, ...]}."""
        try:
            return self._chat(session_id, user_id, message)
        except _EndpointDown as exc:
            raise TimeoutError(f"endpoint unavailable: {exc}") from exc

    def _chat(self, session_id: str, user_id: str, message: str) -> dict:
        message = (message or "").strip()
        history = self._sessions[session_id]
        with self._lock:
            self._update_memory(user_id, message)
        facts = self.get_memory(user_id)

        # Memory-recall questions are answered from the store, without retrieval.
        if RECALL_RE.search(message) and not self._rank_alias(message) and (
            re.search(r"\b(me|my|i)\b", message, re.I)
        ):
            answer = self._answer_recall(message, facts)
            history.append({"user": message, "answer": answer, "citations": [], "query": None})
            return {"answer": answer, "citations": []}

        # Query construction
        rewritten, history_query = None, None
        if self._needs_context(message, history) or (self.rewrite_mode == "always" and history):
            prev = next((t["query"] for t in reversed(history) if t.get("query")), None)
            if prev:
                history_query = f"{prev} {message}"
            if self.rewrite_mode != "off":
                rewritten = self._rewrite(message, history)

        ranked = self.retrieve(message, history_query=history_query, rewritten=rewritten)
        citations: list[str] = []
        if ranked and content_terms(message):
            top = ranked[0][1]
            citations = [pid for pid, s in ranked[: self.max_citations] if s >= self.cite_ratio * top]

        # Back-references ("the three approaches we discussed") pull in earlier turns' papers.
        br = BACKREF.search(message) if history else None
        if br:
            n = next((v for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", message, re.I)), 3)
            prev_pids: list[str] = []
            for t in reversed(history):
                if t.get("primary") and t["primary"] not in prev_pids:
                    prev_pids.append(t["primary"])
                if len(prev_pids) >= n:
                    break
            citations = list(dict.fromkeys(prev_pids[::-1] + citations))[: max(n, self.max_citations)]

        query_for_context = " ".join(x for x in (rewritten, message) if x)
        context = self._build_context(query_for_context, citations) if citations else "(no relevant papers found)"
        messages = [{"role": "system", "content": self._system_prompt(facts)}]
        for t in history[-3:]:
            messages.append({"role": "user", "content": t["user"]})
            messages.append({"role": "assistant", "content": t["answer"][:800]})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"})
        answer = self._complete(messages)

        if citations:
            refs = "; ".join(f"[{i}] {self.papers[p]['title']} ({p})" for i, p in enumerate(citations, 1))
            answer = f"{answer}\n\nSources: {refs}"
        history.append({"user": message, "answer": answer, "citations": citations,
                        "primary": ranked[0][0] if ranked and content_terms(message) else None,
                        "query": rewritten or message})
        del history[:-10]
        return {"answer": answer, "citations": citations}

    def reset_memory(self, user_id: str) -> None:
        """Empty the persistent memory store for this user."""
        with self._lock:
            self._memory[user_id] = []
            self._save_memory()

    def get_memory(self, user_id: str) -> list[str]:
        """Return stored facts about this user as readable strings (read-only view)."""
        return [self._fact_str(f) for f in self._memory.get(user_id, [])]
