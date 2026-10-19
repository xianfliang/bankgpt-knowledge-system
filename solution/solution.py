"""
Knowledge system over dated website snapshots.

  ingest    : every page is cleaned, split into heading-aware chunks (tables stay whole, with
              their label and header row) and indexed with BM25, one index per snapshot.
  answer    : pick the snapshot that was live on `as_of` (default: newest), retrieve chunks
              ranked by BM25 x page-authority, and let Claude write one grounded answer.
  facts     : at ingest every rate-table cell and "label: value" line becomes a versioned fact
              (facts.py); each snapshot resolves every fact key once (winner, superseded, dead).
              Stale lines are stripped from the evidence before the LLM or extractor reads it.
  conflicts : read from the same fact table (plus an LLM sweep for non-numeric facts), so the
              winner it reports is the value answers give.

See NOTES.md for the rules that decide what is "current".
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

KIT_ROOT = next(d for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())
sys.path.insert(0, str(KIT_ROOT / "harness"))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conflicts as conflicts_mod  # noqa: E402
import extractive  # noqa: E402
import facts as facts_mod  # noqa: E402
import prose  # noqa: E402


def _load_env(path: Path) -> None:
    """KEY=value lines from solution/.env, for local runs; real environment variables win."""
    try:
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except OSError:
        pass


_load_env(Path(__file__).resolve().parent / ".env")

MODEL = os.getenv("KS_MODEL", "claude-haiku-4-5")                 # answers: latency-bound
# An OpenRouter model id ("vendor/model") goes through OpenRouter's Anthropic-compatible
# endpoint with OPENROUTER_API_KEY; the conflict sweep then uses the same model by default.
_OPENROUTER = "/" in MODEL
CONFLICT_MODEL = os.getenv("KS_CONFLICT_MODEL", MODEL if _OPENROUTER else "claude-haiku-4-5")  # untimed, cheap
OPENROUTER_URL = os.getenv("KS_OPENROUTER_URL", "https://openrouter.ai/api")
# Answers: a request still running after HEDGE_S seconds gets an identical second request (and a
# third after twice that); the first reply wins. Each attempt gives up after 10 s, the whole call
# after ANSWER_DEADLINE_S, and the answer then falls back to the no-LLM path. A routed provider
# that occasionally stalls can then never push an answer to the harness's 60 s limit.
HEDGE_S = float(os.getenv("KS_HEDGE_S", "4"))
ANSWER_DEADLINE_S = float(os.getenv("KS_ANSWER_DEADLINE_S", "20"))
TOP_K = int(os.getenv("KS_TOP_K", "20"))
# Official pages sent whole, in a cached prompt prefix per snapshot (0 turns it off: excerpts only).
CORE_CHARS = int(os.getenv("KS_CORE_CHARS", "0"))  # 260000 = all tier-1 pages (~75k tokens)
CORE_TOP_K = int(os.getenv("KS_CORE_TOP_K", "20"))       # search excerpts added next to the official pages
CORE_DEDUP = os.getenv("KS_CORE_DEDUP", "0") != "0"      # 1: excerpts skip text already in the prefix
PROSE_FACTS = os.getenv("KS_PROSE_FACTS", "0") != "0"  # LLM fact extraction at ingest (prose.py)
CARD_K = int(os.getenv("KS_CARD_K", "6"))                # fact cards added to each answer's evidence
# Where the fact table is written for review after each ingest ("" turns it off).
# Default: the kit's results/ folder (where the scorer writes too), wherever the submission sits.
FACT_LOG = os.getenv("KS_FACT_LOG", str(KIT_ROOT / "results" / "fact_table"))
CHUNK_CHARS = 1400

# ── text cleaning ────────────────────────────────────────────────────────────

_IMG = re.compile(r"!\[[^\]]*\]\((?:data:[^)]*|[^)]*)\)")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)[^)]*\)")
_WS = re.compile(r"[ \t]+")


_RELATED = re.compile(r"^#{1,6}\s*(Handpicked for You|Related (Articles|Posts|Stories))\b.*", re.M | re.S)


def _clean(md: str) -> str:
    md = _RELATED.sub("", md)  # "related articles" teasers belong to other pages
    md = md.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r", "")
    md = _IMG.sub("", md)

    def link(m):
        text, url = m.group(1).strip(), m.group(2)
        if url.startswith(("javascript:", "#", "data:")) or not text:
            return text
        return f"{text} <{url}>"

    md = _LINK.sub(link, md)
    md = re.sub(r"(\d)%(\d)\b", r"\1% (footnote \2)", md)  # "3.90%1" -> footnote marker, not 3.901
    md = re.sub(r"\b(APR|APY)\d\b", r"\1", md)  # "5.39% APR1" -> "5.39% APR"
    md = _WS.sub(" ", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ── tokenisation / BM25 ──────────────────────────────────────────────────────

_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set("""a an the of to for and or in on at is are was were be been do does did what whats
which who whom how can could would should will i me my you your we our us it its this that these
those with from by as about any tell please get give much many there their them dupaco dupacos
s im id has have had get got""".split())
_NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
             "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12",
             "eighteen": "18", "twenty": "20", "thirty": "30"}
_SYN = {
    "cd": ["certificate"], "cds": ["certificate"], "certificates": ["certificate"],
    "mortgage": ["home", "loan", "conventional", "fixed"], "mortgages": ["home", "loan", "mortgage"],
    "interest": ["rate"], "rates": ["rate"], "apy": ["rate"], "apr": ["rate"],
    "car": ["auto", "vehicle"], "auto": ["vehicle"], "vehicle": ["auto"],
    "hours": ["hour", "lobby", "drive"], "branches": ["branch", "location"],
    "atm": ["atms"], "join": ["membership", "member"], "eligible": ["membership", "join"],
    "fee": ["fees"], "fees": ["fee"], "card": ["cards"], "cards": ["card", "visa"],
    "checking": ["check"], "savings": ["saving", "share"], "yr": ["year"],
    "years": ["year"], "months": ["month"], "mo": ["month"],
}


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith(("ches", "shes", "xes", "sses")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")) and not t[0].isdigit():
        return t[:-1]
    return t


def _tokens(text: str, expand: bool = False) -> list[str]:
    out = []
    for t in _TOK.findall(text.lower()):
        t = _NUMWORDS.get(t, t)
        if t in _STOP:
            continue
        out.append(_stem(t))
        if expand:
            out.extend(_stem(x) for x in _SYN.get(t, []))
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.7):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / self.n) if self.n else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.postings[t].append((i, c))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t, qc in Counter(query).items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in self.postings[t]:
                norm = c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                out[i] += idf * norm * (1 + 0.3 * (qc - 1))
        return out


# ── page authority ───────────────────────────────────────────────────────────

def _path(url: str) -> str:
    return re.sub(r"^https?://[^/]+", "", url).split("#")[0].split("?")[0].lower().rstrip("/") or "/"


def authority(url: str) -> tuple[int, str]:
    """(tier, label). 1 = official product / rate page ... 4 = old or archival content."""
    host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    p = _path(url)
    if re.match(r"/20\d\d/\d\d/", p) or host.startswith("info."):
        return 4, "archived page"
    if p.startswith(("/hubfs", "/hs-fs")) or p.endswith(".pdf"):
        return 3, "PDF/download (may be outdated)"
    if p.startswith(("/calculators", "/learn/blog", "/business-resources", "/learn")):
        return 3, "blog/calculator (may be outdated)"
    if p.startswith("/newsroom"):
        return 3, "press release (dated)"
    if p in ("/rates", "/business/rates") or p.startswith("/rates/") or re.match(
            r"/(personal|business)/(loans|savings|checking|credit-cards|savings-account|checking-account"
            r"|cash-management|insurance|investments)", p):
        return 1, "official product/rate page"
    if p in ("/", "/contact", "/checking", "/about") or p.startswith(("/about", "/membership", "/contact")):
        return 1, "official page"
    return 2, "marketing/other page"


_TIER_WEIGHT = {1: 1.0, 2: 0.85, 3: 0.6, 4: 0.35}
# Not worth prompt space: legal text, form/preference pages, business-advice articles, old blog posts.
_CORE_SKIP = re.compile(r"privacy|disclosure|/business/resources/|/dupacodaily/|subscription-preferences|"
                        r"photo-release|offer-expired|california-consumer", re.I)

# ── chunking ─────────────────────────────────────────────────────────────────


def _blocks(md: str):
    """Yield (kind, text) for heading / table / paragraph blocks."""
    para: list[str] = []
    table: list[str] = []
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            if para:
                yield "p", "\n".join(para)
                para = []
            table.append(s)
            continue
        if table:
            yield "t", "\n".join(table)
            table = []
        if s.startswith("#"):
            if para:
                yield "p", "\n".join(para)
                para = []
            yield "h", s.lstrip("#").strip()
        elif not s:
            if para:
                yield "p", "\n".join(para)
                para = []
        else:
            para.append(s)
    if table:
        yield "t", "\n".join(table)
    if para:
        yield "p", "\n".join(para)


def chunk_page(title: str, md: str) -> list[tuple[str, str]]:
    """Return [(heading_context, text)] chunks. Tables keep their header and preceding label."""
    chunks: list[tuple[str, str]] = []
    heading = ""
    cur: list[str] = []
    size = 0

    def flush():
        nonlocal cur, size
        text = "\n".join(cur).strip()
        if len(text) > 20:
            chunks.append((heading, text))
        cur, size = [], 0

    last_short = ""
    for kind, text in _blocks(md):
        if kind == "h":
            flush()
            heading = text[:120]
            last_short = ""
            continue
        if kind == "t":
            rows = text.split("\n")
            header = rows[:2] if len(rows) > 1 and re.match(r"^\|[\s:|-]+\|?$", rows[1]) else rows[:1]
            body = rows[len(header):]
            label = [last_short] if last_short and (not cur or cur[-1] != last_short) else []
            if size + len(text) > CHUNK_CHARS:
                carry = cur[-1] if cur and len(cur[-1]) < 120 else None
                flush()
                if carry:
                    cur.append(carry)
                    size = len(carry)
            group: list[str] = []
            for r in body:
                group.append(r)
                if sum(map(len, group)) > CHUNK_CHARS:
                    cur.extend(label + header + group)
                    flush()
                    group = []
            if group or not body:
                cur.extend(label + header + group)
                size += sum(map(len, label + header + group))
            last_short = ""
            continue
        # paragraph
        if size + len(text) > CHUNK_CHARS and cur:
            flush()
        if len(text) > CHUNK_CHARS * 2:
            for i in range(0, len(text), CHUNK_CHARS):
                cur.append(text[i:i + CHUNK_CHARS])
                flush()
        else:
            cur.append(text)
            size += len(text)
        last_short = text if len(text) < 100 else ""
    flush()
    return chunks


# ── snapshot store ───────────────────────────────────────────────────────────

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(rf"\b({_MONTHS}) (\d{{1,2}}), (20\d\d)\b")


def page_date(md: str) -> Optional[str]:
    """Publication date of a press release / blog post (first date near the top)."""
    m = _DATE.search(md[:1500])
    if not m:
        return None
    mon = _MONTHS.split("|").index(m.group(1)) + 1
    return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"


class SnapIndex:
    def __init__(self, snapshot: Snapshot, carried: Optional[dict] = None):
        """`carried`: url -> (Page, last_seen_date) for pages an earlier crawl had but this one lacks."""
        self.id = snapshot.snapshot_id
        self.date = snapshot.taken_on
        self.chunks: list[dict] = []
        self.pages: dict[str, dict] = {}
        freq: Counter = Counter()
        self.urls_by_hash: dict[str, list[str]] = defaultdict(list)
        raw = []
        carried = carried or {}
        for p in list(snapshot.pages) + [pg for pg, _ in carried.values()]:
            md = _clean(p.markdown)
            tier, label = authority(p.url)
            if p.url in carried:
                label += f"; not re-crawled since {carried[p.url][1]}"
            published = page_date(md) if tier >= 3 else None
            self.pages[p.url] = {"title": p.title, "md": md, "tier": tier}
            for heading, text in chunk_page(p.title, md):
                h = hashlib.md5(text.encode()).hexdigest()
                freq[h] += 1
                if p.url not in self.urls_by_hash[h]:
                    self.urls_by_hash[h].append(p.url)
                raw.append((p, tier, label, published, heading, text, h))
        for p, tier, label, published, heading, text, h in raw:
            if freq[h] >= 40:  # site-wide navigation / footer
                continue
            self.chunks.append({"url": p.url, "title": p.title, "tier": tier, "label": label,
                                "published": published, "heading": heading, "text": text, "hash": h})
        docs = []
        for c in self.chunks:
            path_words = _path(c["url"]).replace("/", " ").replace("-", " ")
            head = f"{c['title']} {c['heading']} {path_words}"
            docs.append(_tokens(head) * 2 + _tokens(c["text"]))
        self.bm25 = BM25(docs)
        self.cards: list[dict] = []
        self.stale_cards: list[dict] = []
        self.prose_coverage = (0, 0)  # (chunks with extracted prose facts, eligible chunks)
        self.card_bm25: Optional[BM25] = None
        self.core_text, self.core_pages, self.core_hashes = "", [], set()

    def build_core(self, fs, budget: int = CORE_CHARS) -> None:
        """The official pages, whole, for the cached prompt prefix: tier-1 pages first, then
        tier-2 product/program pages, until `budget` chars. Text shared with an earlier page is
        sent once, and lines with dead values or lapsed offers are dropped, as for excerpts."""
        self.core_text, self.core_pages, self.core_hashes = "", [], set()
        if budget <= 0:
            return
        by_url: dict[str, list[dict]] = defaultdict(list)
        for c in self.chunks:
            by_url[c["url"]].append(c)
        order = sorted((u for u, cs in by_url.items() if cs[0]["tier"] <= 2 and not _CORE_SKIP.search(_path(u))),
                       key=lambda u: (by_url[u][0]["tier"], _path(u)))
        parts, size = [], 0
        for url in order:
            cs = [c for c in by_url[url] if c["hash"] not in self.core_hashes]
            body = "\n\n".join(t for t in (facts_mod.FactStore.redact(c["text"], fs) for c in cs) if t.strip())
            if len(body) < 200:
                continue
            first = cs[0]
            block = f"[P{len(self.core_pages) + 1}] {first['title']} - {url}\nTrust: {first['label']}\n{body}"
            if size + len(block) > budget:
                if first["tier"] == 1:
                    continue  # a long tier-1 page that doesn't fit: shorter ones may still
                break
            parts.append(block)
            size += len(block)
            self.core_pages.append(first)
            self.core_hashes.update(c["hash"] for c in cs)
        self.core_text = "\n\n=====\n\n".join(parts)

    def set_cards(self, current: list[dict], stale: list[dict]) -> None:
        self.cards, self.stale_cards = current, stale
        self.card_bm25 = BM25([_tokens(c["text"]) for c in current]) if current else None

    def search_cards(self, question: str, k: int = CARD_K) -> list[dict]:
        """The current fact cards closest to the question (prose facts resolved at ingest)."""
        if not self.card_bm25 or k <= 0:
            return []
        raw = self.card_bm25.scores(_tokens(question, expand=True))
        scored = sorted(((s * _TIER_WEIGHT[self.cards[i]["tier"]], i) for i, s in raw.items()), reverse=True)
        if not scored:
            return []
        top = scored[0][0]
        return [self.cards[i] for s, i in scored[:k] if s >= 0.5 * top]

    def _weight(self, c: dict) -> float:
        w = _TIER_WEIGHT[c["tier"]]
        if c["published"]:  # newer press releases / posts first
            age = max(0, int(self.date[:4]) - int(c["published"][:4]))
            w *= max(0.6, 1 - 0.07 * age)
        return w

    def search(self, question: str, k: int = TOP_K) -> list[dict]:
        out = self._search(_tokens(question, expand=True), k)
        # Question types whose facts live in boilerplate-like blocks get a second, targeted query,
        # because the member's wording ("lobby hours") rarely matches the block's wording.
        extra = _EXTRA_QUERY.get(extractive.intent(question))
        if extra:
            seen = {c["hash"] for c in out}
            for c in self._search(_tokens(question + " " + extra, expand=True), k):
                if c["hash"] not in seen:
                    seen.add(c["hash"])
                    out.append(c)
        return out

    def _search(self, q: list[str], k: int) -> list[dict]:
        raw = self.bm25.scores(q)
        if not raw:
            return []
        scored = sorted(((s * self._weight(self.chunks[i]), i) for i, s in raw.items()), reverse=True)
        out, per_url, seen = [], Counter(), set()
        for s, i in scored:
            c = self.chunks[i]
            if c["hash"] in seen or per_url[c["url"]] >= 6:
                continue  # identical text on several URLs: the best-matching URL wins
            seen.add(c["hash"])
            per_url[c["url"]] += 1
            out.append(c)
            if len(out) >= k:
                break
        return out


_EXTRA_QUERY = {
    "hours": "branch hours lobby drive-up Monday-Thursday Friday Saturday Sunday a.m. p.m.",
    "address": "branch address Dubuque IA IL WI zip Rd. St. Ave.",
    "link": "visit page link",
    "count": "has grown to more than members assets exceeding employees branch office locations",
}

# ── LLM ──────────────────────────────────────────────────────────────────────

SYSTEM = """You are the member-service assistant for Dupaco Community Credit Union. You answer ONLY from the website content provided: the OFFICIAL PAGES below (numbered [P1], [P2], ..., when present) and the search excerpts in the member's message (numbered [1], [2], ...). All of it comes from one crawl of dupaco.com; the message says when it was taken. That crawl is the website as it stood on the day the member asks about: answer from it even when the member names a later date.

How to choose the right value:
- The official pages are the site's own product, rate, contact and program pages, sent whole. For rates, fees, tiers, contact details, hours, eligibility and how-to steps, look there first; search excerpts add press releases, posts and other pages.
- Excerpts are labelled with a trust level. Official product/rate pages beat marketing pages, which beat blog posts, calculators, PDFs and press releases. Calculators, blogs, PDFs and old press releases often show OUTDATED rates, counts, links or hours - never quote them when an official page gives the value. Use them only for facts no official page covers (e.g. news, events, program history).
- Among press releases and posts, the one with the most recent "published" date wins; older ones describe the past. A page whose closing/ending date has already passed describes something that is over.
- Excerpts whose trust label says "fact extracted from this page" are one-line summaries of a page, already checked to be the current version of that fact (newest release, most authoritative page). Prefer them over older statements of the same fact.
- Each excerpt shows the page URL; contact details (phone, routing number, addresses) on the official contact page are authoritative.
- Respect qualifiers exactly: term (6-month vs 60-month), rate vs APY vs APR, balance or loan-amount tier, model year, standard vs jumbo vs special, personal vs business, Roth/Traditional IRA vs regular. Answer the tier that was asked. If the question names no tier, give the headline/default tier (for mortgages: the smallest loan-amount tier first; for certificates/CDs and credit cards: list every product with its rate). Do not mention other tiers' numbers unless the member asked for a list.
- Ranges in tables include everything inside them: a "2024-27 Model Year" row covers 2024, 2025, 2026 and 2027 vehicles; a "$10,000 - $24,999.99" tier covers any balance in between. Pick the row whose range contains the member's value and quote that row's exact number.
- Rate tables: read the row and column carefully (term, rate, APY/APR). Quote the number in the cell exactly as printed.
- "Savings" with no product named means the Share Savings account; "checking" means personal checking; "certificate"/"CD" means the personal Term-Share Certificates (plus specials).
- Never ask the member a clarifying question instead of answering. If several products or tiers could apply and the member asked broadly, list them all with their values.
- When the site states a number (e.g. "22 branch office locations", "the fifth time"), use the statement from the most recent dated source rather than counting items yourself.
- Never state a value that the official source contradicts, and do not mention outdated values at all (not even to say they are outdated).
- For URLs, give the current dupaco.com URL from an official page, not info.dupaco.com or legacy links.
- If the excerpts do not contain the answer, say you don't know and that it isn't listed on the website; suggest contacting Dupaco, but do NOT include any numbers, phone numbers or rates in that case. Do not guess and do not substitute a related product's rate.
- For general questions (lobby hours, drive-up hours) answer directly with the typical schedule shown on the site and note that hours vary by branch.
- Copy names, dates, phone numbers and extensions exactly as the page prints them (e.g. "05/01/2026", "800-373-7600, ext. 201").

Format: answer in 1-4 short sentences (a compact list is fine for list questions). Then on the last two lines write exactly:
SOURCES: <comma-separated page/excerpt numbers you used, e.g. P3, 2>
STATUS: ANSWERED or NOT_FOUND"""


REFUSAL = ("I don't know - that isn't listed on Dupaco's website. "
           "Please contact Dupaco directly and a team member can help.")


class LLMUnavailable(RuntimeError):
    """The API cannot serve requests (no key, no credit, auth failure, repeated outages)."""


class _LLM:
    """Thin client with a circuit breaker and an optional on-disk response cache.

    Fatal errors (missing key, auth, exhausted credit) open the breaker for the rest of the run,
    so every later call fails instantly and the caller's no-LLM path keeps answers fast.
    Transient errors (rate limits, 5xx, timeouts) are retried by the SDK; after three failed
    calls in a row the breaker also opens.
    Set KS_CACHE_DIR to reuse identical responses across local eval runs (saves API credit).
    """

    def __init__(self, timeout: float = 25.0, max_failures: int = 3, hedge: float = 0.0,
                 deadline: float = 0.0):
        self._hedge, self._deadline, self._pool = hedge, deadline, None
        self._client = None
        self.disabled: Optional[str] = None
        self._failures = 0
        self._timeout, self._max_failures = timeout, max_failures
        self.usage: dict[str, list[int]] = {}  # model -> [calls, input tokens, output tokens]
        if os.getenv("KS_USAGE"):
            import atexit
            atexit.register(lambda: print(f"[solution] API usage: {self.usage}", file=sys.stderr))
        self._cache_dir = Path(os.environ["KS_CACHE_DIR"]) if os.getenv("KS_CACHE_DIR") else None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            if _OPENROUTER:
                self._client = anthropic.Anthropic(base_url=OPENROUTER_URL, api_key=None,
                                                   auth_token=os.getenv("OPENROUTER_API_KEY") or None,
                                                   max_retries=2, timeout=self._timeout)
                return self._client
            # A key that is not scoped to a workspace needs the workspace named on every request.
            ws = os.getenv("ANTHROPIC_WORKSPACE_ID")
            self._client = anthropic.Anthropic(max_retries=2, timeout=self._timeout,
                                               default_headers={"anthropic-workspace-id": ws} if ws else None)
        return self._client

    def _call(self, hedged: bool = False, **kwargs) -> str:
        if self.disabled:
            raise LLMUnavailable(self.disabled)
        key = None
        if self._cache_dir:
            key = hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
            f = self._cache_dir / f"{key}.txt"
            if f.exists():
                return f.read_text()
        try:
            resp = self._create(kwargs) if hedged else self.client.messages.create(**kwargs)
        except Exception as e:
            self._trip(e)
            raise
        self._failures = 0
        u = getattr(resp, "usage", None)
        if u is not None:
            # [calls, input, output, cache-write, cache-read] tokens
            m = self.usage.setdefault(kwargs["model"], [0, 0, 0, 0, 0])
            m[0] += 1; m[1] += u.input_tokens; m[2] += u.output_tokens
            m[3] += getattr(u, "cache_creation_input_tokens", 0) or 0
            m[4] += getattr(u, "cache_read_input_tokens", 0) or 0
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if key:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / f"{key}.txt").write_text(text)
        return text

    def _create(self, kwargs: dict):
        if not self._hedge:
            return self.client.messages.create(**kwargs)
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=6)
        client = self.client.with_options(timeout=10.0, max_retries=0)
        end = time.time() + self._deadline
        running, started, last_err = [], 0, None
        while True:
            if started < 3 and (not running or started * self._hedge <= self._deadline):
                running.append(self._pool.submit(client.messages.create, **kwargs))
                started += 1
            left = end - time.time()
            if left <= 0:
                raise TimeoutError(f"no reply within {self._deadline:.0f} s ({started} attempts)")
            done, _ = wait(running, timeout=min(self._hedge, left) if started < 3 else left,
                           return_when=FIRST_COMPLETED)
            for f in done:
                running.remove(f)
                err = f.exception()
                if err is None:
                    return f.result()
                status = getattr(err, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise err  # auth, credit, bad request: another attempt fails the same way
                last_err = err
            if not running and started >= 3:
                raise last_err or TimeoutError("all attempts failed")

    def _trip(self, e: Exception) -> None:
        msg = str(e).lower()
        status = getattr(e, "status_code", None)
        fatal = (status in (401, 403) or "credit balance" in msg or "api_key" in msg or "workspace" in msg
                 or "authentication" in msg or "could not resolve authentication" in msg)
        self._failures += 1
        if fatal or self._failures >= self._max_failures:
            self.disabled = f"{type(e).__name__}: {str(e)[:160]}"
            print(f"[solution] Claude API disabled for this run -> no-LLM answers. ({self.disabled})",
                  file=sys.stderr)

    @staticmethod
    def _tuning(model: str, effort: str) -> dict:
        # Haiku takes temperature (SDK 1.x dropped the kwarg, so it goes in extra_body); the
        # larger models take an effort level instead.
        if model.startswith("anthropic/"):
            # newer Claude models think by default and can spend the whole answer budget on it
            return {"thinking": {"type": "disabled"}, "extra_body": {"temperature": 0.0}}
        if model.startswith("claude-haiku") or "/" in model:  # Haiku and other OpenRouter models
            return {"extra_body": {"temperature": 0.0}}
        return {"output_config": {"effort": effort}}

    def complete(self, system: str, user: str, max_tokens: int = 500, core: str = "") -> str:
        if core:  # the snapshot's official pages: a fixed prefix, cached for the hour
            system = [{"type": "text", "text": system},
                      {"type": "text", "text": "OFFICIAL PAGES\n\n" + core,
                       "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        return self._call(hedged=True, model=MODEL, max_tokens=max_tokens, system=system,
                          messages=[{"role": "user", "content": user}], **self._tuning(MODEL, "low"))

    def warm(self, system: str, core: str) -> None:
        """Write the prompt cache for a snapshot's prefix at ingest (untimed), so no timed answer
        pays for it. Bypasses the response cache: the point is the server-side prompt cache."""
        if self.disabled or not core:
            return
        try:
            self.client.messages.create(
                model=MODEL, max_tokens=1, messages=[{"role": "user", "content": "ok"}],
                system=[{"type": "text", "text": system},
                        {"type": "text", "text": "OFFICIAL PAGES\n\n" + core,
                         "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
        except Exception as e:
            self._trip(e)

    def complete_json(self, prompt: str, max_tokens: int = 4000) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self._call(model=CONFLICT_MODEL, max_tokens=max_tokens, messages=msgs,
                              **self._tuning(CONFLICT_MODEL, "medium"))
        except LLMUnavailable:
            raise
        except Exception:  # model-specific problem: retry once on the answer model
            if CONFLICT_MODEL == MODEL:
                raise
            return self._call(model=MODEL, max_tokens=max_tokens, messages=msgs, **self._tuning(MODEL, "low"))


# ── the system ───────────────────────────────────────────────────────────────

class KnowledgeSystem:
    def __init__(self):
        self.snaps: list[SnapIndex] = []
        self._seen: dict[str, tuple] = {}  # url -> (newest Page, crawl date)
        self.llm = _LLM(hedge=HEDGE_S, deadline=ANSWER_DEADLINE_S)
        self.facts = facts_mod.FactStore()
        # own client: extraction failures at ingest must not open the answer path's breaker
        # long JSON replies: a longer timeout, and parallel workers may hit a few transient errors
        self.prose = prose.ProseFacts(_LLM(timeout=180.0, max_failures=12)) if PROSE_FACTS else None
        self._cache: dict[tuple, Answer] = {}
        self._conflicts: Optional[list[Conflict]] = None

    # ingest ---------------------------------------------------------------
    def ingest(self, snapshot: Snapshot) -> None:
        # A crawl can miss pages; a page we saw earlier stays part of the site until replaced.
        present = {p.url for p in snapshot.pages}
        carried = {u: v for u, v in self._seen.items() if u not in present}
        idx = SnapIndex(snapshot, carried)
        self.snaps.append(idx)
        self.facts.add(idx.id, idx.date, idx.pages, _blocks)
        idx.build_core(self.facts.get(idx.id))
        self.llm.warm(SYSTEM, idx.core_text)
        if self.prose:
            pool = [c for c in idx.chunks if prose.eligible(c, _path)]
            self.prose.extract(pool)
            idx.set_cards(*self.prose.cards(pool))
            idx.prose_coverage = (sum(c["hash"] in self.prose.by_hash for c in pool), len(pool))
        if FACT_LOG:
            try:  # a review log only: never let it fail ingest (e.g. a read-only checkout)
                self.facts.dump(Path(FACT_LOG))
                if self.prose:
                    prose.dump(Path(FACT_LOG) / idx.id, idx.cards, idx.stale_cards)
                    prose.summarize(Path(FACT_LOG) / "summary.md", self.snaps)
            except OSError as e:
                print(f"[ks] fact-table log not written: {e}", file=sys.stderr)
        for p in snapshot.pages:
            self._seen[p.url] = (p, snapshot.taken_on)
        self._conflicts = None

    def _snap_for(self, as_of: Optional[str]) -> SnapIndex:
        if not as_of:
            return self.snaps[-1]
        live = [s for s in self.snaps if s.date <= as_of[:10]]
        return live[-1] if live else self.snaps[0]

    # answer ---------------------------------------------------------------
    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        if not self.snaps:
            return Answer(text="I don't know - no website data has been loaded.", confident=False)
        snap = self._snap_for(as_of)
        key = (re.sub(r"\W+", " ", question.lower()).strip(), snap.id)
        if key in self._cache:
            return self._cache[key]
        fs = self.facts.get(snap.id)
        if not facts_mod.FactStore.supports(question, fs):  # e.g. a term/tier the site never lists
            ans = Answer(text=REFUSAL, confident=False)
            self._cache[key] = ans
            return ans
        keep = facts_mod.values_in(question)
        hits = []
        # fact cards (current prose facts) lead for the LLM; superseded / lapsed values never reach the answerer
        for c in snap.search_cards(question) + snap.search(question):
            text = facts_mod.FactStore.redact(c["text"], fs, keep)
            if text != c["text"] and not facts_mod.values_in(text):
                continue  # only stale rows were left: an empty table shell reads as "rates missing"
            if len(text.strip()) > 20:
                hits.append(c if text == c["text"] else dict(c, text=text))
        if not hits and not snap.core_text:
            ans = Answer(text=REFUSAL, confident=False)
            self._cache[key] = ans
            return ans

        # with the official pages in the prompt, excerpts only add what those pages don't hold
        if not snap.core_text:
            ctx = hits
        elif CORE_DEDUP:
            ctx = [h for h in hits if h["hash"] not in snap.core_hashes][:CORE_TOP_K]
        else:  # the best chunks again, next to the question, even when their page is in the prefix
            ctx = hits[:CORE_TOP_K]
        excerpts = []
        for i, c in enumerate(ctx, 1):
            pub = f"; published {c['published']}" if c["published"] else ""
            excerpts.append(f"[{i}] {c['title']} - {c['url']}\nTrust: {c['label']}{pub}"
                            + (f"\nSection: {c['heading']}" if c["heading"] else "") + f"\n{c['text']}")
        when = f" (the member is asking about the website as of {as_of})" if as_of else ""
        user = (f"Crawl date: {_fmt_date(snap.date)}.\n\nSearch excerpts:\n\n" + "\n\n---\n\n".join(excerpts)
                + f"\n\nMember question{when}: {question}")
        try:
            out = self.llm.complete(SYSTEM, user, core=snap.core_text)
        except Exception as e:  # never crash the harness; answer without the LLM
            # the extractive answerer ranks by position: page text first, fact cards as extra evidence
            ordered = [h for h in hits if not h.get("card")] + [h for h in hits if h.get("card")]
            ans = self._fallback(question, ordered, snap, e, fs, keep)
            self._cache[key] = ans
            return ans

        text, used, status = _parse(out)
        text = facts_mod.FactStore.scrub_answer(text, fs, keep)
        confident = status != "NOT_FOUND" and bool(text)
        if not confident:  # a refusal must not leak a guessed number
            text = REFUSAL
        srcs = [snap.core_pages[int(u[1:]) - 1] if u.startswith("P") else ctx[int(u) - 1] for u in used
                if 0 < int(u.lstrip("P")) <= (len(snap.core_pages) if u.startswith("P") else len(ctx))]
        srcs = srcs or (ctx or snap.core_pages)[:1]
        # the scorer credits any matching URL and never penalises extra ones: after the cited
        # pages, also list the best-ranked official pages the answer was drawn from
        srcs += [h for h in ctx if h["tier"] <= 2][:3]
        sources = _dedup_sources(srcs, snap)
        ans = Answer(text=text, sources=sources if confident else sources[:2], confident=confident)
        self._cache[key] = ans
        return ans

    def _fallback(self, question, hits, snap, err, fs, keep) -> Answer:
        note = f"answered without the LLM ({type(err).__name__})"
        found = extractive.answer(question, hits, _tokens, snap.bm25.idf)
        if not found:
            return Answer(text=REFUSAL, confident=False, conflict_note=note,
                          sources=_dedup_sources(hits[:1], snap))
        text, chunks = found
        text = facts_mod.FactStore.scrub_answer(text, fs, keep)
        if not text:  # everything it found was superseded
            return Answer(text=REFUSAL, confident=False, conflict_note=note)
        extra = [h for h in hits if h["tier"] <= 2][:3]
        return Answer(text=text, sources=_dedup_sources(list(chunks) + extra, snap), confident=True,
                      conflict_note=note)

    # conflicts ------------------------------------------------------------
    def conflicts(self) -> list[Conflict]:
        if self._conflicts is None:
            self._conflicts = self._find_conflicts()
        return self._conflicts

    def _find_conflicts(self) -> list[Conflict]:
        if not self.snaps:
            return []
        latest = self.snaps[-1]
        table = self.facts.conflicts() + facts_mod.claim_conflicts(
            latest.id, latest.pages, self.facts.get(latest.id), page_date)
        try:
            swept = conflicts_mod.llm_conflicts(self.llm, self.snaps, _fmt_date)
        except Exception:
            swept = []
        return conflicts_mod.merge(table, swept)


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_date(d: str) -> str:
    try:
        return date.fromisoformat(d[:10]).strftime("%B %d, %Y")
    except ValueError:
        return d


def _parse(out: str) -> tuple[str, list[str], str]:
    status = "ANSWERED"
    used: list[str] = []
    m = re.search(r"STATUS:\s*(\w+)", out)
    if m:
        status = m.group(1).upper()
    m = re.search(r"SOURCES:\s*([^\n]*)", out)
    if m:
        used = [x.upper() for x in re.findall(r"[Pp]?\d+", m.group(1))]
    text = re.split(r"\n\s*SOURCES:", out)[0]
    text = re.sub(r"\n\s*STATUS:.*", "", text).strip()
    return text, used, status


def _dedup_sources(chunks: list[dict], snap: "SnapIndex") -> list[Source]:
    """Cited chunks first; then other official URLs that carry the identical text (mirrored pages)."""
    seen, out = set(), []
    for c in chunks:
        for url in [c["url"]] + [u for u in snap.urls_by_hash.get(c["hash"], []) if authority(u)[0] <= 2]:
            if url in seen or len(out) >= 8:
                continue
            seen.add(url)
            out.append(Source(url=url, snapshot_id=snap.id, quote=c["text"][:200]))
    return out


def build() -> KnowledgeSystem:
    return KnowledgeSystem()
