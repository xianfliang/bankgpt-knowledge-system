"""
Conflict detection. Runs once, lazily, when `conflicts()` is first called (not timed).

Two complementary detectors:

  1. Fact table (deterministic, facts.py): numeric facts whose versions disagree, read from
     the same table the answer path uses.
  2. LLM sweep (non-table facts such as branch counts, routing numbers, links, award counts,
     hours). For a fixed list of member-facing topics we retrieve official and low-authority
     excerpts from the newest snapshot and ask Claude to list contradictions, as JSON.

Both feed one list; entries that name the same pair of values are merged.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from contract import Conflict, Source


# Rate tables that the fact table (facts.py) reads cell by cell are left out: there it is
# exact, and sweep findings on them were mostly the same contradiction again or a tier mix-up,
# which lowers conflict precision.
TOPICS = [
    "motorcycle, boat and RV loan rates",
    "personal loan APR by term",
    "home equity loan and line of credit rates",
    "Share Savings dividend rates and APY tiers",
    "money market account rates",
    "Bright Savings online savings account rate",
    "health savings account (HSA) rates",
    "checking account dividend rates (1st Rate Checking)",
    "Platinum Visa and Rewards Visa credit card APR",
    "business credit card APR",
    "business savings and business certificate rates",
    "routing number",
    "phone number and contact information",
    "number of branches / branch office locations",
    "number of members, employees and total assets",
    "lobby hours",
    "drive-up hours",
    "branch closures and branch addresses",
    "membership eligibility and charter area",
    "how to join online (join link)",
    "link to rates page or certificate rates page",
    "branch and ATM locator link",
    "Forbes and other awards and recognition",
    "fees (lost or replacement debit card, overdraft, wire transfer)",
    "first-time homebuyer grant and down payment assistance amount",
    "switch and earn / new member bonus offer",
    "number of surcharge-free ATMs",
    "volunteer hours and volunteer time off statistics",
]

SWEEP_PROMPT = """You audit a credit union website for self-contradictions. Below are excerpts from the CURRENT crawl of the site (taken {snap_date}); each has its URL and a trust label. Topic: {topic}

List every pair of excerpts that state DIFFERENT values for the SAME fact (same product, same term, same tier/model year/balance range, same rate type). Typical real contradictions: a calculator, blog, PDF, landing page or older press release still shows an outdated rate, count, link, phone/routing number or hours that the official page has since changed.

Also contradictions: an expired or old promotion that is still published with a different value for the same product, and an "OLDER CRAWL" excerpt whose value differs from the current page (the value changed over time).

NOT contradictions: different terms/tiers/products; dividend rate vs APY; APR vs interest rate; a currently running introductory rate vs the standard rate; purchase vs refinance; personal vs business; values that are clearly about different years or events. Also not contradictions: the same value written differently (20.5% vs 20.50%, dash spacing), or a footnote digit glued to a value ($3001 = $300 plus footnote 1).

For each contradiction decide which value is current: official product/rate/contact pages beat calculators, blogs, PDFs and landing pages; a newer dated press release beats an older one.

Return ONLY a JSON array (possibly empty). Each element:
{{"topic": "<short name of the fact, include product and term/tier, e.g. '60-month auto loan APR (2024-27 model year)'>",
  "current": "<current value, short and verbatim with its unit, e.g. '5.39%' or '22 branch office locations' or 'join.dupaco.com'>",
  "stale": ["<each outdated value, same style>"],
  "current_url": "<url>", "stale_urls": ["<url>"],
  "reason": "<one sentence>"}}

Excerpts:
{excerpts}"""


# ── LLM sweep ─────────────────────────────────────────────────────────────

def _older_versions(snaps, hits, topic: str) -> list[dict]:
    """The same official pages as they read in older crawls, when their text differs."""
    out, cur_text = [], {c["hash"] for c in hits}
    for old in snaps[:-1]:
        for c in old.search(topic, k=8):
            if c["tier"] <= 2 and c["hash"] not in cur_text and any(h["url"] == c["url"] for h in hits):
                cur_text.add(c["hash"])
                out.append(dict(c, label=f"OLDER CRAWL of {old.date} (superseded by the current crawl)"))
    return out[:6]


def _sweep_topic(llm, snaps, topic: str, fmt_date) -> list[Conflict]:
    snap = snaps[-1]
    hits = snap.search(topic, k=18)
    extra = [c for c in snap.search(topic, k=60) if c["tier"] >= 3 and c not in hits][:8]
    chunks = hits + extra + _older_versions(snaps, hits, topic)
    if not chunks:
        return []
    excerpts = "\n\n---\n\n".join(
        f"[{i}] {c['url']}\nTrust: {c['label']}" + (f"; published {c['published']}" if c["published"] else "")
        + f"\n{c['text']}" for i, c in enumerate(chunks, 1))
    prompt = SWEEP_PROMPT.format(snap_date=fmt_date(snap.date), topic=topic, excerpts=excerpts)
    try:
        raw = llm.complete_json(prompt)
    except Exception:
        return []
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            current = str(it["current"]).strip()
            stale = [str(s).strip() for s in it.get("stale", []) if str(s).strip() and str(s).strip() != current]
        except (KeyError, TypeError, AttributeError):
            continue
        if not current or not stale:
            continue
        urls = [str(u).split()[0] for u in [it.get("current_url")] + list(it.get("stale_urls") or [])
                if u and str(u).strip()]
        out.append(Conflict(
            topic=str(it.get("topic") or topic)[:120],
            values=[current] + stale,
            sources=[Source(url=u, snapshot_id=snaps[-1].id) for u in dict.fromkeys(urls)],
            resolution=current,
            reason=str(it.get("reason") or "official page beats outdated page")[:300]))
    return out


def llm_conflicts(llm, snaps, fmt_date) -> list[Conflict]:
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda t: _sweep_topic(llm, snaps, t, fmt_date), TOPICS))
    # at most two findings per topic: a long list for one topic is mostly near-duplicates
    return [c for r in results for c in r[:2]]


# ── merge ────────────────────────────────────────────────────────────────────

def _vals(c: Conflict) -> frozenset:
    out = set()
    for v in c.values:
        n = re.findall(r"\d[\d,]*(?:\.\d+)?", v)
        out.add(float(n[0].replace(",", "")) if n else v.lower())
    return frozenset(out)


def _nums(v: str) -> tuple:
    v = re.sub(r"\(.*", "", v)  # drop the model's parenthetical commentary
    return tuple(n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", v))


def _same_number(a: str, b: str) -> bool:
    """20.5 == 20.50, and "3001" is "300" with a glued footnote marker."""
    if float(a) == float(b):
        return True
    lo, hi = sorted((a, b), key=len)
    return "." not in hi and len(hi) == len(lo) + 1 and hi.startswith(lo)


def _trivial(c: Conflict) -> bool:
    """A cheaper sweep model sometimes reports formatting differences as contradictions:
    the same numbers written differently are not a conflict."""
    base = _nums(c.values[0])
    if not base:
        return False
    for v in c.values[1:]:
        n = _nums(v)
        if len(n) != len(base) or not all(_same_number(x, y) for x, y in zip(n, base)):
            return False
    return True


def merge(table: list[Conflict], swept: list[Conflict]) -> list[Conflict]:
    """Table conflicts are kept as-is; an LLM finding is dropped when it is only a formatting
    difference, or when an earlier entry already names two of its values (the same
    contradiction found twice)."""
    out = list(table)
    seen = [_vals(c) for c in table]
    for c in swept:
        if _trivial(c):
            continue
        v = _vals(c)
        if any(len(v & sv) >= 2 or v == sv for sv in seen):
            continue
        seen.append(v)
        out.append(c)
    return out
