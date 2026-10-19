"""
Versioned fact table, compiled at ingest (no LLM).

Every rate-table cell and every short "label: value" line becomes a Fact keyed by
(normalised label, kind). Each snapshot then resolves every key once:

  winner      the value shown by the best-authority, non-expired facts of that key
              (several different values at the best tier = ambiguous, no winner)
  superseded  values the same key showed on a lower-authority page, on an expired promotion,
              or on the official page in an older crawl
  dead        superseded values that are no key's winner in this snapshot

The answer path uses `dead` to strip stale lines from the evidence before anything reads it,
and `supports()` to refuse questions about a term/tier the site never lists. `conflicts()` is
read from the same table, so the winner it reports is the value answers give.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

from contract import Conflict, Source

_PCT = re.compile(r"(\d{1,2}\.\d{1,3})\s?%")
_SKIP_COL = re.compile(r"point|payment|min|balance|term|deposit|fee", re.I)
_VALUE = re.compile(r"(?<![\w.$])(\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand)\b)?"
                    r"|\d{1,2}(?:\.\d{1,3})?\s?%)", re.I)
_NUMVAL = re.compile(r"(\$)?\s*(\d[\d,]*(?:\.\d+)?)\s*(%|million|billion|thousand)?", re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
# table cells that describe a requirement, not the product: never part of a row label
_NOT_LABEL_COL = re.compile(r"payment|point|min|max|to open|to earn", re.I)
_FOOTNOTE = re.compile(r"[*\\]|\d$")
_EXAMPLE = re.compile(r"\byou(?:'ll| will)? (?:would )?pay\b|\bpayment example|\bmonthly payment|\btotal (?:interest|cost)", re.I)
_EMPTY_CELL = re.compile(r"^(none|n/?a|-+|—|–)$", re.I)

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_EXPIRY = re.compile(
    r"\b(?:ended|expired|ends|expires|valid (?:through|until)|good through|through|until)\s+(?:on\s+)?"
    rf"(?P<d>\d{{4}}-\d{{2}}-\d{{2}}|(?:{_MONTHS})[a-z]*\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})",
    re.I)

_NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "eighteen": "18",
             "twenty": "20", "thirty": "30"}
_TERM = re.compile(r"\b(\d{1,3})\s*-?\s*(months?|mos?|years?|yrs?)\b", re.I)
# Words that make a different product: a question naming one needs a fact that names it too.
QUALIFIERS = ("jumbo", "roth", "traditional", "ira", "refinanc", "special", "hsa", "health savings",
              "student", "holiday", "business", "money market", "home equity", "boat", "motorcycle", "rv")
_RATE_Q = re.compile(r"\b(rates?|aprs?|apys?|interest|dividends?|yields?)\b", re.I)


def num(v: str) -> Optional[tuple]:
    """'3.50%' -> ('%', 3.5); '$2.50' -> ('$', 2.5). Same value spelled differently compares equal."""
    m = _NUMVAL.search(v or "")
    if not m:
        return None
    suffix = (m.group(3) or "").lower()
    unit = "$" if m.group(1) else ("%" if suffix == "%" else "")
    return unit, round(float(m.group(2).replace(",", "")) * _SCALE.get(suffix, 1), 4)


def values_in(text: str) -> set:
    return {n for v in _VALUE.findall(text or "") if (n := num(v))}


def norm_label(label: str) -> str:
    s = label.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"\\|\*|_", "", s)
    return re.sub(r"\s+", " ", s).strip(" |:-")


def _iso(d: str) -> Optional[str]:
    d = d.strip().rstrip(".")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", d)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.fullmatch(rf"({_MONTHS})[a-z]*\.? (\d{{1,2}})(?:st|nd|rd|th)?,? (\d{{4}})", d, re.I)
    if m:
        mon = _MONTHS.split("|").index(m.group(1).lower()[:3]) + 1
        return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"
    return None


def expired(text: str, today: str) -> bool:
    """The text says its offer ended / runs through a date that has already passed."""
    for m in _EXPIRY.finditer(text or ""):
        d = _iso(m.group("d"))
        if d and d < today[:10]:
            return True
    return False


def _terms(text: str) -> set:
    """Terms in months: '6-month' -> 6, '1 year' / '1-Year' -> 12."""
    t = re.sub(r"\b(" + "|".join(_NUMWORDS) + r")\b", lambda m: _NUMWORDS[m.group(1).lower()], text, flags=re.I)
    out = set()
    for n, unit in _TERM.findall(t):
        out.add(int(n) * (12 if unit.lower().startswith("y") else 1))
    return out


@dataclass(frozen=True)
class Fact:
    key: str        # normalised label
    kind: str       # APY / APR / rate / % / $
    value: str      # as printed, e.g. "3.50%"
    label: str      # as printed, for display
    url: str
    tier: int
    title: str      # page title: picks the canonical URL among mirrored pages
    context: str    # title / heading / column header: used for term and qualifier checks
    expired: bool


@dataclass
class FactSnapshot:
    id: str
    date: str
    facts: list = field(default_factory=list)
    by_key: dict = field(default_factory=lambda: defaultdict(list))   # (key, kind) -> [Fact]
    winners: dict = field(default_factory=dict)                        # (key, kind) -> (value, tier, url)
    dead: set = field(default_factory=set)                             # {(unit, value)}
    superseded: dict = field(default_factory=lambda: defaultdict(set))  # (key, kind) -> {value}
    official_urls: dict = field(default_factory=lambda: defaultdict(set))  # (key, kind) -> {tier-1 url}
    retired: dict = field(default_factory=dict)  # key the official page dropped -> its successor key


# ── extraction ───────────────────────────────────────────────────────────────

def _table_facts(url, page, blocks, today):
    context = ""
    for kind_b, text in blocks:
        if kind_b in ("h", "p"):
            if len(text) < 80:
                context = text
            continue
        rows = [r for r in text.split("\n") if r.startswith("|")]
        if len(rows) < 2:
            continue
        header = [c.strip().lower() for c in rows[0].strip("|").split("|")]
        col_kinds = [_col_kind(h) for h in header]
        for r in rows[1:]:
            if re.match(r"^\|[\s:|-]+\|?$", r):
                continue
            cells = [c.strip() for c in r.strip("|").split("|")]
            label_cells = [c for j, c in enumerate(cells)
                           if c and not _PCT.search(c) and not re.match(r"^\$?[\d,.\s]+$", c)
                           and not _EMPTY_CELL.match(c)
                           and not (j < len(header) and _NOT_LABEL_COL.search(header[j]))]
            if not label_cells:
                continue
            label = " | ".join(label_cells)
            if not re.search(r"[a-z]{3}", label.lower()):
                continue
            if not re.search(r"month|year|savings|checking|certificate|visa|market|loan|account|special"
                             r"|ira|roth|traditional|fixed|conventional|equity", label.lower()):
                label = f"{context} | {label}" if context else label
            gone = expired(context, today) or expired(r, today)
            for j, c in enumerate(cells):
                m = _PCT.search(c)
                if not m or float(m.group(1)) == 0:
                    continue
                col = header[j] if j < len(header) else ""
                if _SKIP_COL.search(col) and not re.search(r"rate|apy|apr|yield", col):
                    continue
                kind = "APR" if "APR" in c and not col_kinds[j] else (col_kinds[j] if j < len(col_kinds) else None)
                if not kind:
                    continue
                if col_kinds.count(kind) > 1:  # keep the columns apart
                    kind = f"{kind} ({norm_label(_FOOTNOTE.sub('', col))})"
                yield Fact(norm_label(label), kind, m.group(1) + "%", label, url, page["tier"], page["title"],
                           f"{page['title']} {context} {col}", gone)


def _line_facts(url, page, blocks, today):
    """Short lines / sentences that state exactly one rate or amount: 'Lost card fee: $2.50'."""
    heading = ""
    for kind_b, text in blocks:
        if kind_b == "h":
            heading = text
            continue
        if kind_b != "p":
            continue
        for line in text.split("\n"):
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line.strip(" -*")):
                if len(sent) > 200:
                    continue
                vals = _VALUE.findall(sent)
                if len(vals) != 1:
                    continue
                m = _VALUE.search(sent)
                label = re.sub(r"\s*<[^>]*>", "", sent[:m.start()]).strip(" :-–—")
                if len(label) < 6 or len(label) > 110 or not re.search(r"[A-Za-z]{3}", label):
                    continue
                if _EXAMPLE.search(label):  # a calculator's worked example, not a published fact
                    continue
                low = sent.lower()
                kind = ("APY" if "apy" in low or "yield" in low else "APR" if "apr" in low
                        else "$" if vals[0].startswith("$") else "%")
                # the sentence often omits the product ("variable APR of"); the heading names it
                if heading and norm_label(heading) not in norm_label(label):
                    label = f"{heading} | {label}"
                yield Fact(norm_label(label), kind, vals[0].replace(" ", ""), label, url, page["tier"],
                           page["title"], f"{page['title']} {heading}", expired(sent, today))


def extract(pages: dict, block_iter: Callable, today: str) -> list:
    out = []
    for url, page in pages.items():
        blocks = list(block_iter(page["md"]))
        out.extend(_table_facts(url, page, blocks, today))
        out.extend(_line_facts(url, page, blocks, today))
    return out


# ── the store ────────────────────────────────────────────────────────────────

class FactStore:
    def __init__(self):
        self.snaps: list[FactSnapshot] = []

    def add(self, snap_id: str, date: str, pages: dict, block_iter: Callable) -> FactSnapshot:
        fs = FactSnapshot(snap_id, date)
        fs.facts = extract(pages, block_iter, date)
        for f in fs.facts:
            fs.by_key[(f.key, f.kind)].append(f)
            if f.tier == 1 and not f.expired:
                fs.official_urls[(f.key, f.kind)].add(f.url)
        self._retire(fs, pages)

        superseded: set = set()
        for k, facts in fs.by_key.items():
            if k in fs.retired:
                fs.superseded[k] |= {f.value for f in facts}
                succ = fs.retired[k]
                if succ:
                    fs.superseded[succ] |= {f.value for f in facts}
                continue
            live = [f for f in facts if not f.expired]
            for f in facts:
                if f.expired:
                    fs.superseded[k].add(f.value)
            if not live:
                continue
            best = min(f.tier for f in live)
            top = Counter(f.value for f in live if f.tier == best)
            if len(top) > 1:
                continue  # two equally trusted pages disagree: no safe winner
            (win, _), = top.most_common(1)
            fs.winners[k] = (win, best, _canonical([f for f in live if f.tier == best and f.value == win]))
            for f in live:
                if f.tier > best and num(f.value) != num(win):
                    fs.superseded[k].add(f.value)
            # an official value that changed since an earlier crawl
            if best == 1:
                for old in self.snaps:
                    w = old.winners.get(k)
                    if w and w[1] == 1 and num(w[0]) != num(win):
                        fs.superseded[k].add(w[0])

        alive = {num(w[0]) for w in fs.winners.values()}
        for vals in fs.superseded.values():
            superseded |= {num(v) for v in vals}
        fs.dead = {v for v in superseded if v and v not in alive}
        self.snaps.append(fs)
        return fs

    def _retire(self, fs: FactSnapshot, pages: dict) -> None:
        """A row an official page showed in an earlier crawl but no longer shows (while the page
        itself is still there) is retired: other pages still quoting it are stale. Its successor
        is the official row on that page with the same words and the most numbers in common
        ('2023-26 Model Year | 60 Months' -> '2024-27 Model Year | 60 Months')."""
        official_now = {u for u, p in pages.items() if p["tier"] == 1}
        labels_now = {k[0] for k in fs.official_urls}
        for k in fs.by_key:
            if k[0] in labels_now:
                continue
            for old in reversed(self.snaps):
                # any kind: a calculator may call the same row's APR an "APY"
                urls = set().union(*(u for k2, u in old.official_urls.items() if k2[0] == k[0])) & official_now
                if urls:
                    fs.retired[k] = _successor(k, urls, fs)
                    break

    # ── review log ───────────────────────────────────────────────────────────

    def status(self, f: Fact, fs: FactSnapshot) -> str:
        k = (f.key, f.kind)
        if k in fs.retired:
            return "retired row"
        if f.expired:
            return "expired offer"
        w = fs.winners.get(k)
        if not w:
            return "ambiguous (equal-authority pages disagree)"
        if num(f.value) == num(w[0]):
            return "current" if f.tier == w[1] else "agrees with current"
        return "stale"

    def dump(self, out_dir: Path) -> None:
        """Write the fact table for human review:
             <snapshot>/facts.csv  every fact with its status and the key's current value
             timeline.csv          each official key's current value per crawl
             summary.md            counts, dead values, retired rows, conflicts"""
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            fs = self.snaps[-1]
            d = out_dir / fs.id
            d.mkdir(exist_ok=True)
            with open(d / "facts.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["status", "label", "kind", "value", "current_value", "dead_value", "tier",
                            "url", "successor_of_retired", "context", "key"])
                order = sorted(fs.facts, key=lambda f: (f.key, f.kind, f.tier, f.url))
                for f in order:
                    k = (f.key, f.kind)
                    win = fs.winners.get(k, ("", 0, ""))[0]
                    succ = fs.retired.get(k)
                    w.writerow([self.status(f, fs), f.label, f.kind, f.value, win,
                                "yes" if num(f.value) in fs.dead else "", f.tier, f.url,
                                f"{succ[0]} {succ[1]}" if succ else "", f.context.strip(), f.key])

            keys = sorted({k for s in self.snaps for k, w in s.winners.items() if w[1] == 1})
            with open(out_dir / "timeline.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["label", "kind", "changed"] + [s.id for s in self.snaps] + ["official url"])
                for k in keys:
                    vals = [s.winners[k][0] if k in s.winners and s.winners[k][1] == 1 else "" for s in self.snaps]
                    label = next(s.by_key[k][0].label for s in reversed(self.snaps) if k in s.by_key)
                    url = next(s.winners[k][2] for s in reversed(self.snaps) if k in s.winners)
                    changed = len({num(v) for v in vals if v}) > 1
                    w.writerow([label, k[1], "yes" if changed else ""] + vals + [url])

            lines = ["# Fact table review", "",
                     "Built at ingest by `facts.py`. Tier 1 = official product/rate page ... 4 = archived.", "",
                     "| Snapshot | Facts | Keys | Current values | Dead values | Retired rows | Expired |",
                     "|---|---|---|---|---|---|---|"]
            for s in self.snaps:
                lines.append(f"| {s.id} | {len(s.facts)} | {len(s.by_key)} | {len(s.winners)} | {len(s.dead)} "
                             f"| {len({k[0] for k in s.retired})} | {sum(f.expired for f in s.facts)} |")
            lines += ["", f"## Dead values in {fs.id}", "",
                      "Stale everywhere they appear, and no key's current value: stripped from answers.", "",
                      "| Value | Where it still appears |", "|---|---|"]
            for v in sorted(fs.dead, key=lambda x: (x[0], x[1])):
                where = sorted({f"{f.label} ({f.url})" for f in fs.facts if num(f.value) == v})
                shown = ("<br>".join(where[:4]) + (f"<br>… +{len(where) - 4} more" if len(where) > 4 else "")
                         or "(no longer on any page; stale in an earlier crawl)")
                lines.append(f"| {v[0] == '$' and '$' or ''}{v[1]:g}{v[0] == '%' and '%' or ''} | {shown} |")
            lines += ["", f"## Retired rows in {fs.id}", "",
                      "Rows an official page used to show and no longer does.", "",
                      "| Retired row | Successor | Still quoted on |", "|---|---|---|"]
            for k, succ in sorted(fs.retired.items()):
                urls = sorted({f.url for f in fs.by_key[k]})
                lines.append(f"| {fs.by_key[k][0].label} ({k[1]}) | {succ[0] + ' ' + succ[1] if succ else '-'} "
                             f"| {'<br>'.join(urls)} |")
            conf = self.conflicts()
            lines += ["", f"## Conflicts from the fact table ({len(conf)})", "",
                      "| Topic | Winner | Other values | Reason |", "|---|---|---|---|"]
            for c in conf:
                lines.append(f"| {c.topic} | {c.resolution} | {', '.join(c.values[1:])} | {c.reason} |")
            lines += ["", "Full rows: `<snapshot>/facts.csv`. Value history: `timeline.csv`.", ""]
            (out_dir / "summary.md").write_text("\n".join(lines))
        except OSError as e:  # the log is a convenience; never fail ingest over it
            print(f"[facts] could not write the fact log: {e}", file=sys.stderr)

    def get(self, snap_id: str) -> Optional[FactSnapshot]:
        return next((s for s in self.snaps if s.id == snap_id), None)

    # ── answer-side helpers ────────────────────────────────────────────────

    @staticmethod
    def redact(text: str, fs: FactSnapshot, keep: set = frozenset()) -> str:
        """Drop lines that state a dead value (or a lapsed offer with a value)."""
        out = []
        for line in text.split("\n"):
            vals = values_in(line) - keep
            if vals & fs.dead:
                continue
            if vals and expired(line, fs.date):
                continue
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def scrub_answer(text: str, fs: FactSnapshot, keep: set = frozenset()) -> str:
        """Safety net on generated text: remove sentences that quote a dead value."""
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        kept = [p for p in parts if not ((values_in(p) - keep) & fs.dead)]
        return text if len(kept) == len(parts) else " ".join(kept).strip()

    @staticmethod
    def supports(question: str, fs: FactSnapshot) -> bool:
        """A rate question naming a term (and product qualifiers like 'jumbo') needs at least one
        live fact with that term and those qualifiers; otherwise the site does not list it."""
        if not _RATE_Q.search(question):
            return True
        terms = _terms(question)
        if not terms:
            return True
        q = question.lower()
        quals = [w for w in QUALIFIERS if re.search(rf"\b{w}", q)]
        for f in fs.facts:
            if f.expired or (f.key, f.kind) in fs.retired:
                continue
            ctx = f"{f.label} {f.context} {f.url}".lower()
            if terms & _terms(ctx) and all(re.search(rf"\b{w}", ctx) for w in quals):
                return True
        return False

    # ── conflicts ────────────────────────────────────────────────────────────

    def conflicts(self) -> list[Conflict]:
        if not self.snaps:
            return []
        latest = self.snaps[-1]
        out: list[Conflict] = []
        keys: list[tuple] = []  # the fact key each entry is about (parallel to `out`)
        covered = set()

        # (a) newest crawl: a lower-authority or expired page disagrees with the winner
        for k, (win, best, win_url) in latest.winners.items():
            facts = latest.by_key[k]
            losers = {}
            for f in facts:
                if num(f.value) != num(win) and (f.tier > best or f.expired):
                    losers.setdefault(f.value, f)
            if not losers:
                continue
            covered.add(k)
            keys.append(k)
            label = facts[0].label
            why = []
            for v, f in losers.items():
                why.append(f"{f.url} shows {v}" + (" for an offer that has ended" if f.expired else
                                                    " (lower-authority page, outdated)"))
            out.append(Conflict(
                topic=f"{label} {k[1]}",
                values=[win] + list(losers),
                sources=[Source(url=win_url, snapshot_id=latest.id)]
                        + [Source(url=f.url, snapshot_id=latest.id) for f in losers.values()],
                resolution=win,
                reason=f"{win_url} is the most authoritative current page; " + "; ".join(why)))

        # (a2) a row the official page dropped or renamed, still quoted elsewhere
        for k, succ in latest.retired.items():
            if not succ or succ not in latest.winners or succ in covered:
                continue
            win, _, win_url = latest.winners[succ]
            stale = {f.value: f for f in latest.by_key[k] if num(f.value) != num(win)}
            if not stale:
                continue
            covered.add(succ)
            keys.append(succ)
            old_label, new_label = latest.by_key[k][0].label, latest.by_key[succ][0].label
            out.append(Conflict(
                topic=f"{new_label} {succ[1]}",
                values=[win] + list(stale),
                sources=[Source(url=win_url, snapshot_id=latest.id)]
                        + [Source(url=f.url, snapshot_id=latest.id) for f in stale.values()],
                resolution=win,
                reason=f"{win_url} now lists '{new_label}' at {win}; "
                       + ", ".join(f.url for f in stale.values())
                       + f" still shows the retired row '{old_label}'"))

        # (b) an official value changed between crawls (newest wins)
        for k, (win, best, win_url) in latest.winners.items():
            if best != 1 or k in covered:
                continue
            hist = []
            for s in self.snaps:
                w = s.winners.get(k)
                if w and w[1] == 1 and (not hist or num(hist[-1][2]) != num(w[0])):
                    hist.append((s.date, s.id, w[0], w[2]))
            if len(hist) < 2:
                continue
            values = list(dict.fromkeys([hist[-1][2]] + [v for _, _, v, _ in reversed(hist[:-1])]))
            keys.append(k)
            out.append(Conflict(
                topic=f"{latest.by_key[k][0].label} {k[1]}",
                values=values,
                sources=[Source(url=u, snapshot_id=sid) for _, sid, _, u in reversed(hist)],
                resolution=win,
                reason="value changed between crawls (" + ", ".join(f"{d}: {v}" for d, _, v, _ in hist)
                       + "); the newest crawl is current"))

        # rate and APY of the same row move together: keep the APY/APR entry only
        has_yield = {_strip_kind(c.topic) for c in out if c.topic.endswith(("APY", "APR"))}
        kept = [(k, c) for k, c in zip(keys, out)
                if c.topic.endswith(("APY", "APR")) or _strip_kind(c.topic) not in has_yield]
        return self._grouped(kept, latest)

    @staticmethod
    def _grouped(entries: list, latest: FactSnapshot) -> list[Conflict]:
        """One entry per contradiction, not per table cell. Rows of the same official page with
        the same term (the 30-year mortgage at four loan amounts, purchase and refinance; the
        17-month special, jumbo and IRA versions) come from one stale page or one rate change,
        so they are reported together: every value involved, and the current value of each row
        (rate and APR/APY) in the resolution. A long list of near-identical entries would only
        lower precision."""
        groups: dict[tuple, list] = {}
        for k, c in entries:
            win_url = c.sources[0].url if c.sources else ""
            terms = frozenset(_terms(latest.by_key[k][0].label)) if latest.by_key.get(k) else frozenset()
            gk = (win_url, terms) if terms else (win_url, k)
            groups.setdefault(gk, []).append((k, c))
        out = []
        for members in groups.values():
            if len(members) == 1:
                out.append(members[0][1])
                continue
            current, stale, rows, reasons, sources, seen_src = [], [], [], [], [], set()
            for k, c in members:
                current.append(c.resolution)
                stale += [v for v in c.values if num(v) != num(c.resolution)]
                label = latest.by_key[k][0].label
                win_url = c.sources[0].url if c.sources else ""
                kinds = [(kind, latest.winners[(k[0], kind)][0]) for kind in ("rate", "APR", "APY")
                         if (k[0], kind) in latest.winners and latest.winners[(k[0], kind)][2] == win_url]
                rows.append(f"{label}: " + " / ".join(f"{v} {kind}" for kind, v in kinds or [(k[1], c.resolution)]))
                reasons.append(c.reason)
                for src in c.sources:
                    if src.url not in seen_src:
                        seen_src.add(src.url)
                        sources.append(src)
            labels = list(dict.fromkeys(latest.by_key[k][0].label for k, _ in members))
            win_url = members[0][1].sources[0].url if members[0][1].sources else ""
            title = next((f.title for k, _ in members for f in latest.by_key[k] if f.url == win_url), "")
            title = (title or "").split("|")[0].strip()
            out.append(Conflict(
                topic=(f"{title}: " if title else "") + "; ".join(labels[:6]) + (" ..." if len(labels) > 6 else ""),
                values=list(dict.fromkeys(current + stale)),
                sources=sources,
                resolution="; ".join(rows),
                reason=f"{len(members)} rows of the same table: " + " | ".join(dict.fromkeys(reasons))))
        return out


def _col_kind(col: str) -> Optional[str]:
    if _SKIP_COL.search(col) and not re.search(r"rate|apy|apr|yield", col):
        return None
    if "apy" in col or "yield" in col:
        return "APY"
    if "apr" in col:
        return "APR"
    if "rate" in col or "dividend" in col or "interest" in col:
        return "rate"
    return None


def _words(text: str) -> set:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z]+", text.lower())}


def _canonical(facts: list) -> str:
    """Several URLs serve the same page (crawl mirrors such as /home-improvement-loan/ showing the
    Home Loans page). Cite the one whose path matches the page title best, then the shortest."""
    def score(f):
        path = _words(re.sub(r"^https?://[^/]+", "", f.url).split("#")[0]) - {"personal", "business", "www"}
        title = _words(f.title)
        return (-(len(path & title) - len(path - title)), len(f.url), f.url)
    return min(facts, key=score).url


def _successor(k: tuple, urls: set, fs: FactSnapshot) -> Optional[tuple]:
    words = re.findall(r"[a-z]+", k[0])
    nums = set(re.findall(r"\d+", k[0]))
    best, best_n = None, -1
    for k2, u2 in fs.official_urls.items():
        if not (u2 & urls) or re.findall(r"[a-z]+", k2[0]) != words:
            continue
        n = len(nums & set(re.findall(r"\d+", k2[0]))) + 0.5 * (k2[1] == k[1])
        if n >= 1 and n > best_n:
            best, best_n = k2, n
    return best


def _strip_kind(topic: str) -> str:
    return re.sub(r"\s(APY|APR|rate|%|\$)$", "", topic)


# ── dated claims in prose ────────────────────────────────────────────────────
# Press-release boilerplate restates a few facts on every release ("... 22 branch office
# locations"), so older releases keep old values. These facts are not in any rate table; the
# newest published statement is current. Kept to patterns the site repeats verbatim, so a
# match is the same fact, not a different count that happens to share words.
_CLAIMS = [
    ("Number of Dupaco branch office locations", re.compile(r"\b(\d{1,2}) branch office locations\b", re.I)),
    ("How many times Forbes has recognized Dupaco",
     re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) time Forbes recognized\b",
                re.I)),
]
# "earns you up to 3.00% APY" on a promo page, above anything the official table pays
_UP_TO = re.compile(r"up to \**(\d+(?:\.\d+)?)%\** APY", re.I)
_PRODUCTS = ("checking", "money market", "savings")


def claim_conflicts(snap_id: str, pages: dict, fs: FactSnapshot, published: Callable) -> list[Conflict]:
    """`pages`: url -> {title, md, tier} of the newest crawl; `published(md)` -> 'YYYY-MM-DD' or None."""
    out = []
    for topic, rx in _CLAIMS:
        seen: dict[str, tuple] = {}  # value -> (newest date stating it, url)
        for url, p in pages.items():
            date = published(p["md"]) or ""
            for m in rx.finditer(p["md"]):
                v = m.group(1).lower()
                if v not in seen or date > seen[v][0]:
                    seen[v] = (date, url)
        dated = {v: d for v, d in seen.items() if d[0]}
        if len(dated) < 2:
            continue
        order = sorted(dated, key=lambda v: dated[v][0], reverse=True)
        win = order[0]
        out.append(Conflict(
            topic=topic, values=order,
            sources=[Source(url=dated[v][1], snapshot_id=snap_id) for v in order],
            resolution=win,
            reason=f"the newest release ({dated[win][0]}, {dated[win][1]}) says {win}; older ones say "
                   + ", ".join(f"{v} ({dated[v][0]})" for v in order[1:])))
    # promo pages promising more than the official rate table pays for that product
    for prod in _PRODUCTS:
        official = [num(w[0])[1] for (k, kind), w in fs.winners.items()
                    if kind == "APY" and w[1] == 1 and prod in k and num(w[0])]
        if not official:
            continue
        top = max(official)
        for url, p in pages.items():
            if p["tier"] == 1:
                continue
            for sent in re.split(r"(?<=[.!?])\s+", p["md"]):
                m = _UP_TO.search(sent)
                if not m or prod not in sent.lower() or float(m.group(1)) <= top:
                    continue
                if prod == "savings" and "money market" in sent.lower():
                    continue
                out.append(Conflict(
                    topic=f"Highest {prod} APY", values=[f"{top:.2f}%", f"{float(m.group(1)):.2f}%"],
                    sources=[Source(url=url, snapshot_id=snap_id)],
                    resolution=f"{top:.2f}%",
                    reason=f"{url} promises up to {m.group(1)}% APY on {prod}; the official rate table's "
                           f"highest {prod} APY is {top:.2f}%"))
                break
            else:
                continue
            break
    return out
