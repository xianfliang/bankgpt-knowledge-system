"""
Prose facts, extracted by Claude at ingest (untimed).

Tables and "label: value" lines are parsed without an LLM (facts.py). What they miss sits in
prose: who is eligible, which mobile wallets work, how many times an award was won, volunteer
counts per year, branch hours and closures. Each unique chunk of an official page or press
release is sent to Claude once (a chunk that is unchanged in a later crawl reuses its facts),
and every fact becomes a short "fact card":

    Membership eligibility (charter area): people who live or work in counties of Iowa, ...

Cards with the same (subject, attribute, qualifier, year) are resolved per snapshot like the
numeric facts: the most authoritative page wins, then the most recently published one. Only
current cards are searchable, so an older press release's "fourth time" card never competes
with the newer "fifth time" card.
"""
from __future__ import annotations

import hashlib
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

BATCH_CHARS = 12000
WORKERS = 8

PROMPT = """You extract facts from excerpts of a credit union's website (Dupaco) so that a member-service assistant can answer questions later.

For every excerpt, list the facts a member might ask about: rates and fees with their tier, amounts and limits, eligibility rules, lists of options (e.g. which mobile wallets or card types), counts (branches, members, employees, volunteer hours, how many times an award was won), dates (openings, closures, deadlines, effective dates), hours, addresses, phone and routing numbers, links, and what a program or product offers.

Rules:
- Only facts stated in the excerpt. Copy numbers, names and list items exactly; give complete lists.
- subject: the product, program, branch, award or organization, named as the site names it (e.g. "Dupaco", "Share Savings", "Forbes Best-in-State Credit Unions", "Asbury branch").
- attribute: a short noun phrase (e.g. "eligibility", "supported mobile wallets", "times recognized", "employees who used volunteer time off", "lobby hours", "APY").
- qualifier: tier, term, branch or condition that makes the value specific, else "".
- year: the year the fact is about if the excerpt says so, else "".
- value: one self-contained sentence fragment with the answer, e.g. "people who live or work in counties of Iowa, Illinois and Wisconsin in the charter area, or have a family member who is a member".
- Skip navigation, legal boilerplate, generic advice and marketing slogans. At most 8 facts per excerpt; none is fine.

Return ONLY a JSON array: [{{"id": <excerpt number>, "subject": "", "attribute": "", "qualifier": "", "year": "", "value": ""}}]

Excerpts:
{excerpts}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def eligible(chunk: dict, path: Callable) -> bool:
    return chunk["tier"] <= 2 or path(chunk["url"]).startswith("/newsroom")


def _batches(chunks: list[dict]):
    cur, size = [], 0
    for c in chunks:
        if cur and size + len(c["text"]) > BATCH_CHARS:
            yield cur
            cur, size = [], 0
        cur.append(c)
        size += len(c["text"])
    if cur:
        yield cur


def _extract_batch(llm, batch: list[dict]) -> dict[str, list[dict]]:
    excerpts = "\n\n---\n\n".join(
        f"[{i}] {c['title']} ({c['url']})" + (f"; published {c['published']}" if c["published"] else "")
        + (f"\nSection: {c['heading']}" if c["heading"] else "") + f"\n{c['text']}"
        for i, c in enumerate(batch, 1))
    out: dict[str, list[dict]] = {c["hash"]: [] for c in batch}
    try:
        raw = llm.complete_json(PROMPT.format(excerpts=excerpts), max_tokens=6000)
    except Exception:
        return {}  # not cached: retried by the next ingest
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    for it in items if isinstance(items, list) else []:
        try:
            i = int(it["id"])
            value = str(it["value"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= i <= len(batch) or not value:
            continue
        out[batch[i - 1]["hash"]].append({k: str(it.get(k) or "").strip()[:300]
                                          for k in ("subject", "attribute", "qualifier", "year", "value")})
    return out


class ProseFacts:
    def __init__(self, llm):
        self.llm = llm
        self.by_hash: dict[str, list[dict]] = {}   # chunk hash -> extracted facts (reused across crawls)
        self.failed = 0                              # chunks whose batch failed (no prose facts this run)

    def extract(self, chunks: list[dict]) -> None:
        todo, seen = [], set()
        for c in chunks:
            if c["hash"] not in self.by_hash and c["hash"] not in seen:
                seen.add(c["hash"])
                todo.append(c)
        if not todo:
            return
        if getattr(self.llm, "disabled", None):
            self.failed += len(todo)
            print(f"[prose] API unavailable: {len(todo)} chunks get no prose facts", file=sys.stderr)
            return
        batches = list(_batches(todo))
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for b, res in zip(batches, pool.map(lambda b: _extract_batch(self.llm, b), batches)):
                self.by_hash.update(res)
                self.failed += 0 if res else len(b)
        missing = sum(1 for c in todo if c["hash"] not in self.by_hash)
        print(f"[prose] {len(todo) - missing}/{len(todo)} new chunks extracted"
              + (f"; {missing} failed (retried next ingest)" if missing else ""), file=sys.stderr)

    def cards(self, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
        """(current cards, superseded cards) for one snapshot's chunks."""
        groups: dict[tuple, list] = {}
        for c in chunks:
            for f in self.by_hash.get(c["hash"], []):
                key = (_norm(f["subject"]), _norm(f["attribute"]), _norm(f["qualifier"]), f["year"])
                groups.setdefault(key, []).append((f, c))
        current, stale = [], []
        for key, items in groups.items():
            # best authority first, then the most recently published page, then the URL (stable)
            items.sort(key=lambda fc: (fc[1]["tier"], _neg_date(fc[1]["published"]), fc[1]["url"]))
            win_f, win_c = items[0]
            seen_urls = set()
            for f, c in items:
                card = _card(f, c)
                if _norm(f["value"]) == _norm(win_f["value"]):
                    if c["url"] not in seen_urls:
                        seen_urls.add(c["url"])
                        current.append(card)
                else:
                    stale.append(dict(card, superseded_by=win_c["url"], current_value=win_f["value"]))
        return current, stale


def _neg_date(d) -> str:
    """Sort key: newer dates first, undated last."""
    if not d:
        return "~"
    return "".join(chr(ord("9") - ord(ch) + ord("0")) if ch.isdigit() else ch for ch in d)


def _card(f: dict, c: dict) -> dict:
    qual = f" ({f['qualifier']})" if f["qualifier"] else ""
    year = f" [{f['year']}]" if f["year"] else ""
    text = f"{f['subject']} - {f['attribute']}{qual}{year}: {f['value']}"
    return {"url": c["url"], "title": c["title"], "tier": c["tier"], "published": c["published"],
            "label": c["label"] + "; fact extracted from this page", "heading": f["subject"][:120],
            "text": text, "hash": hashlib.md5(("card " + text + c["url"]).encode()).hexdigest(),
            "card": True, "fact": f}


def dump(out_dir: Path, current: list[dict], stale: list[dict]) -> None:
    """<snapshot>/prose_facts.csv: every extracted prose fact, current or superseded."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "prose_facts.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["status", "subject", "attribute", "qualifier", "year", "value", "tier", "published",
                        "url", "superseded_by", "current_value"])
            rows = [("current", c) for c in current] + [("stale", c) for c in stale]
            rows.sort(key=lambda r: (_norm(r[1]["fact"]["subject"]), _norm(r[1]["fact"]["attribute"]), r[0]))
            for status, c in rows:
                f = c["fact"]
                w.writerow([status, f["subject"], f["attribute"], f["qualifier"], f["year"], f["value"],
                            c["tier"], c["published"] or "", c["url"], c.get("superseded_by", ""),
                            c.get("current_value", "")])
    except OSError as e:
        print(f"[prose] could not write the prose fact log: {e}", file=sys.stderr)


def summarize(summary: Path, snaps: list) -> None:
    """Append prose-fact coverage per snapshot to summary.md (a gap means failed API batches)."""
    try:
        lines = ["", "## Prose facts (LLM extraction at ingest)", "",
                 "| Snapshot | Chunks extracted | Current facts | Superseded facts |", "|---|---|---|---|"]
        for s in snaps:
            done, total = s.prose_coverage
            pct = f" ({100 * done // total}%)" if total else ""
            lines.append(f"| {s.id} | {done}/{total}{pct} | {len(s.cards)} | {len(s.stale_cards)} |")
        lines += ["", "Rows: `<snapshot>/prose_facts.csv`.", ""]
        with open(summary, "a") as fh:
            fh.write("\n".join(lines))
    except OSError as e:
        print(f"[prose] could not extend the summary: {e}", file=sys.stderr)
