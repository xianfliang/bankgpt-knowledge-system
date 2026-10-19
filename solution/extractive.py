"""
Evidence-assembling answerer (no LLM). Used when the Claude API is unavailable, and as the
precision layer that tells the LLM prompt what kind of answer is wanted.

Design: classify the question's *answer type*, then pull that kind of fact out of every retrieved
chunk (not just the single best sentence) and assemble a short answer from the best few units:

  rate     -> the matching table row (term / model-year / balance tier / qualifiers), or for broad
              "all cards / certificates" questions every row of the best table
  hours    -> the Lobby / Drive-up schedule blocks (several distinct ones when asked "same hours?")
  address  -> street line of the named branch
  link     -> best official URL (anchor text + path overlap), never info.* or staging hosts
  date     -> sentences carrying a date, newest dated source first
  count    -> the "N ..." statement from the newest dated source
  general  -> top sentences, expanded to their bullet list / paragraph

It refuses when a proper noun in the question never appears on the site (e.g. an unknown branch)
or when the evidence does not cover the question's terms.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

REFUSE = ("", [])  # sentinel: a typed extractor decided the site does not say

QUALIFIERS = {"jumbo", "roth", "traditional", "business", "ira", "refinance", "purchase", "special",
              "rewards", "platinum", "student", "holiday", "bright", "health", "hsa", "money", "market",
              "achieve", "investor", "kid", "teen", "club", "equity", "boat", "rv", "motorcycle", "used",
              "introductory", "intro", "transfer", "promotional", "cash", "advance"}
QUALIFIERS = {q[:6] if len(q) > 6 else q for q in QUALIFIERS}  # tokens are compared on 6-letter prefixes
CHATTER = {"repeat", "tell", "give", "provid", "charge", "offer", "want", "need", "know", "show", "say", "pleas",
           "right", "current", "curren", "today", "latest", "good", "best", "wonder", "look", "find", "help"}
RATEVOCAB = {"rate", "apr", "apy", "intere", "loan", "annual", "percen", "dividend", "yield", "charge", "accoun", "year"}
RATE_Q = re.compile(r"\b(rates?|aprs?|apys?|interest|dividends?|yields?|percent)\b", re.I)
YEAR_RANGE = re.compile(r"\b(20\d\d)\s*[-–]\s*(\d{2,4})\b")
MONEY_RANGE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*[-–]\s*\$\s?([\d,]+(?:\.\d+)?)")

# ── question intent ──────────────────────────────────────────────────────────

_I = re.I
INTENTS = [
    ("hours", re.compile(r"\b(hours|lobby|drive[- ]?up|open(?:s|ing)? (?:at|on)|close[sd]? (?:at|on))\b", _I)),
    ("address", re.compile(r"\b(address|located|location of|where is|street)\b", _I)),
    ("link", re.compile(r"\b(link|url|website|web page|webpage|site)\b|\bwhere (?:do|can|should) i (?:go|find|see)\b", _I)),
    ("count", re.compile(r"\bhow many\b", _I)),
    ("date", re.compile(r"\b(when|what (?:date|day|year)|still open|still available|close[sd]?|closing|closure|open today)\b", _I)),
]
BROAD = re.compile(r"\b(all|every|each|which|what (?:credit )?cards|credit cards|certificates?|cds?|rates|"
                   r"list|options|rates for)\b", _I)
GENERIC = {"dupaco", "what", "when", "where", "how", "who", "which", "can", "does", "do", "is", "are",
           "i", "my", "the", "a", "an", "and", "or", "if", "please", "tell", "give", "get", "provide",
           "we", "you", "your", "it", "in", "of", "on", "to", "for", "any", "would", "like", "know"}


def intent(question: str) -> str:
    for name, rx in INTENTS:
        if rx.search(question):
            return name
    return "rate" if RATE_Q.search(question) else "general"


def _unknown_entities(question: str, idf: dict, tokens: Callable) -> list[str]:
    """Capitalised words (not sentence-initial) that appear nowhere on the site."""
    words = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{3,}\b", question)
    return [w for w in words if w.lower() not in GENERIC and not any(t in idf for t in tokens(w))
            and w.lower() not in ("october", "september", "november", "december", "january", "february",
                                  "march", "april", "june", "july", "august", "monday", "tuesday",
                                  "wednesday", "thursday", "friday", "saturday", "sunday", "visa")]


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_value(c: str) -> bool:
    return "%" in c or bool(re.fullmatch(r"\$\s?[\d,]+(?:\.\d+)?", c.strip()))


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _clean_cell(c: str) -> str:
    return re.sub(r"\s*\(footnote \d+\)|\\|\*", "", c).strip()


def _official(c: dict) -> bool:
    return c["tier"] <= 2


def _pub(c: dict) -> str:
    return c.get("published") or "0000-00-00"


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _strip_links(t: str) -> str:
    return re.sub(r"\s*<https?://[^>]*>", "", t)


# ── rate rows ────────────────────────────────────────────────────────────────

def _row_score(row_cells, context: str, qtok: set, qnums: list[float], tokens: Callable) -> float:
    label = " ".join(c for c in row_cells if not _is_value(c))
    rtok = set(tokens(label + " " + context))
    s = 2.0 * len(qtok & rtok)
    for n in qnums:
        if re.search(rf"(?<![\d.]){int(n)}(?![\d])", label):
            s += 3
        for a, b in YEAR_RANGE.findall(label):
            lo = int(a)
            hi = int(b) if len(b) == 4 else int(a[:2] + b)
            if lo <= n <= hi:
                s += 4
        for a, b in MONEY_RANGE.findall(label):
            if _num(a) <= n <= _num(b):
                s += 4
    s -= 2.5 * len((rtok & QUALIFIERS) - qtok)
    s += 1.5 * len((qtok & QUALIFIERS) & rtok)
    return s


def _rows(c: dict):
    """Yield (cells, header, context) for each data row of every table in the chunk."""
    header, context = None, f"{c['heading']} {c['title']}"
    for ln in c["text"].split("\n"):
        if not ln.startswith("|"):
            header = None
            if len(ln) < 80:
                context = f"{ln} {c['heading']}"
            continue
        cells = [_clean_cell(x) for x in ln.strip("|").split("|")]
        if re.fullmatch(r"[\s:|-]*", ln.replace("|", "")):
            continue
        if header is None:
            header = cells
            continue
        if any("%" in x for x in cells):
            yield cells, header, context


def _row_text(cells, header) -> Optional[str]:
    parts = []
    for j, x in enumerate(cells):
        h = header[j] if j < len(header) else ""
        if "%" in x and not re.search(r"payment|min|point", h, re.I):
            parts.append(f"{h} {x}".strip())
    label = ", ".join(x for x in cells if x and not _is_value(x))
    return f"{label}: {'; '.join(parts)}." if parts else None


def best_row(question: str, hits: list[dict], tokens: Callable) -> Optional[tuple]:
    qtok = set(tokens(question, True))
    qdistinct = qtok - RATEVOCAB - CHATTER
    qnums = [_num(x) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", question) if _num(x) >= 1]
    named_amount = any(n >= 1000 and not 1990 < n < 2100 for n in qnums)
    best = None
    for rank, c in enumerate(hits):
        for cells, header, context in _rows(c):
            ov = _row_score(cells, context, qtok, qnums, tokens)
            label = " ".join(x for x in cells if not _is_value(x))
            distinct = len(qdistinct & set(tokens(label + " " + context)))
            s = ov + 3 * (3 - c["tier"]) - 0.15 * rank
            if not named_amount:  # prefer the headline (smallest "or less") tier
                m = re.search(r"\$([\d,]+) or less", " ".join(cells))
                s -= _num(m.group(1)) / 1e6 if m else 0
            if best is None or s > best[2]:
                text = _row_text(cells, header)
                if text:
                    best = (text, c, s, ov, distinct)
    return best


def table_rows(question: str, hits: list[dict], tokens: Callable, limit: int = 8):
    """Broad rate question: the rows most related to the question from official chunks, one per label."""
    qtok = set(tokens(question, True))
    scored, seen = [], set()
    for rank, c in enumerate(hits):
        if c["tier"] > 2:
            continue
        for cells, header, context in _rows(c):
            text = _row_text(cells, header)
            if not text:
                continue
            rtok = set(tokens(text + " " + context))
            ov = len(qtok & rtok) - 1.5 * len((rtok & QUALIFIERS) - qtok) * 0
            key = _norm(text)
            if key in seen or ov <= 0:
                continue
            seen.add(key)
            scored.append((ov + 0.5 * (3 - c["tier"]) - 0.1 * rank, text, c))
    scored.sort(key=lambda x: -x[0])
    return scored[:limit]


def rate_facts(question: str, hits: list[dict], tokens: Callable, idf: dict, limit: int = 8):
    """Broad rate question: table rows AND bullet/sentence rate statements, labelled by their heading."""
    qtok = set(tokens(question, True))
    key = _key_tokens(set(tokens(question)) - RATEVOCAB, idf, _proper_tokens(question, tokens))
    out = [(sc, t, c) for sc, t, c in table_rows(question, hits, tokens, limit * 2)]
    seen = {_norm(t) for _, t, _ in out}
    for rank, c in enumerate(hits):
        if c["tier"] > 2:
            continue
        for sent, cc, r, para in _units([c]):
            if "%" not in sent or not re.search(r"\b(APR|APY|rate|dividend)\b", sent, re.I):
                continue
            label = c["heading"] if c["heading"] and len(c["heading"]) < 60 else c["title"]
            if len(sent) > 230:
                continue  # footnote / legal text, not a rate statement
            text = f"{label}: {_strip_links(sent)}"
            htok = set(tokens(label + " " + sent))
            ov = len(qtok & htok) + 0.5 * (3 - c["tier"]) - 0.1 * rank
            if len(qtok & set(tokens(label))) == 0:
                ov -= 1.5
            if _norm(text) in seen or ov <= 0.5:
                continue
            seen.add(_norm(text))
            out.append((ov, text, c))
    if key:  # every fact must mention the product the member named (else "construction loan" gets auto rates)
        top = max(key, key=lambda t: idf.get(t, 0))
        out = [f for f in out if top in set(tokens(f[1] + " " + f[2]["title"] + " " + f[2]["heading"]))]
    elif not out:
        return []
    out.sort(key=lambda x: -x[0])
    return out[:limit]


def near_rows(question, hits, tokens, best):
    """Rows of the best row's table whose score is within 4 points of it (the product family)."""
    qtok = set(tokens(question, True))
    out, seen = [], set()
    for cells, header, context in _rows(best[1]):
        sc = _row_score(cells, context, qtok, [], tokens)
        if sc >= best[3] - 4:
            t = _row_text(cells, header)
            if t and _norm(t) not in seen:
                seen.add(_norm(t))
                out.append((sc, t))
    out.sort(key=lambda x: -x[0])
    return [(t, best[1]) for _, t in out[:5]]


def sibling_rows(chunk: dict, best_text: str, tokens: Callable):
    """Other tiers of the same product (same label apart from the amount) with a different rate."""
    base = re.sub(r"\s*\$[\d,]+( or less)?.*$", "", best_text.split(": Rate")[0].split(": APR")[0])
    outs, seen = [], {_norm(best_text.split(": ", 1)[-1])}
    for cells, header, _ in _rows(chunk):
        text = _row_text(cells, header)
        if not text or not text.startswith(base):
            continue
        body = _norm(text.split(": ", 1)[-1])
        if body in seen:
            continue
        seen.add(body)
        outs.append(text)
    return outs[:2]


# ── text units ───────────────────────────────────────────────────────────────

def _units(hits: list[dict]):
    """(text, chunk, rank, paragraph) for every sentence / bullet / line."""
    for rank, c in enumerate(hits):
        for para in re.split(r"\n\s*\n", c["text"]):
            if para.lstrip().startswith("|"):
                continue
            para = para.strip()
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])|\n+", para):
                sent = _strip_links(sent).strip(" -|*#")
                if len(sent) < 20 or sent.count("|") > 2 or sent.endswith("?"):
                    continue
                yield sent, c, rank, para


def _proper_tokens(question: str, tokens: Callable) -> set:
    words = re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][\w$-]{2,}\b", question)
    return {t for w in words if w.lower() not in GENERIC for t in tokens(w)}


def _key_tokens(qtok: set, idf: dict, proper: set = frozenset()) -> set:
    """The question's most distinctive tokens (e.g. 'forbes', 'galena', 'tart'): a good unit has them."""
    ranked = sorted((t for t in qtok if t in idf and t not in CHATTER), key=lambda t: -idf[t])
    return set(ranked[:max(1, (len(ranked) + 1) // 2)][:3]) | (set(proper) & qtok)


def _unit_score(sent, c, rank, qtok, idf, tokens, wants_number=False, bonus=0.0, key=None) -> float:
    stok = set(tokens(sent))
    ov = sum(idf.get(t, 1.0) for t in qtok & stok)
    if not ov:
        return 0.0
    path = re.sub(r"^https?://[^/]+", "", c["url"]).replace("/", " ").replace("-", " ")
    ctx = set(tokens(c["title"] + " " + c["heading"] + " " + path))
    ov += 0.5 * sum(idf.get(t, 1.0) for t in qtok & ctx - stok)
    s = ov / math.sqrt(len(stok) + 6) + 0.6 * (3 - c["tier"]) - 0.03 * rank + bonus
    if key:
        have = (len(key & stok) + 0.5 * len((key & ctx) - stok)) / len(key)
        s *= 0.35 + 0.65 * have
    if wants_number and re.search(r"\d", sent):
        s += 0.5
    if c["published"]:  # among dated posts the newest one is the current one
        s += 0.12 * max(0, int(c["published"][:4]) - 2015)
    return s


_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
           "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
DATE_RX = re.compile(rf"\b(?:{_MONTHS})\.? \d{{1,2}}(?:st|nd|rd|th)?(?:, \d{{4}})?|\b(?:{_MONTHS})\.? \d{{4}}\b"
                     rf"|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b|\bby the end of (?:the )?\w+\b", re.I)
ADDR_RX = re.compile(r"\b\d{2,6} [A-Z0-9][\w .]*?(?:St|Street|Rd|Road|Ave|Avenue|Ln|Lane|Dr|Drive|Blvd|Hwy|Highway|"
                     r"Pkwy|Ct|Court|Way|Pl|Place)\b\.?,? [\w .]+, [A-Z]{2},? \d{5}")
COUNT_RX = re.compile(r"\b\d[\d,]*\+?\s+(?:[A-Za-z-]+\s+){0,3}(?:branch(?:es)?|locations?|employees|members|offices|"
                      r"ATMs?|times?|years?)\b|\b(?:first|second|third|fourth|fifth|sixth|seventh)\b", re.I)
TIME_RX = re.compile(r"\d{1,2}(?::\d{2})?\s?[ap]\.?m\.?", re.I)


def _expand_block(para: str, limit: int = 700) -> str:
    return para if len(para) <= limit else para[:limit].rsplit(" ", 1)[0] + "…"


# ── typed extractors ─────────────────────────────────────────────────────────

_HOURS_BLOCK = re.compile(
    r"(?P<kind>\*{0,2}(?:Lobby|Drive[- ]?up)(?: hours)?\*{0,2}:?)\s*\n?"
    r"(?P<body>(?:\s*(?:Monday|Mon|Tuesday|Tue|Wednesday|Wed|Thursday|Thu|Friday|Fri|Saturday|Sat|Sunday|Sun)"
    r"[^\n]*(?:\n|$)){1,7})", re.I)


def hours_blocks(hits: list[dict], want: str):
    """Distinct (kind, schedule, chunk) blocks, newest official/dated first."""
    found, seen = [], set()
    for rank, c in enumerate(hits):
        for m in _HOURS_BLOCK.finditer(c["text"]):
            kind = "drive-up" if "drive" in m.group("kind").lower() else "lobby"
            body = " ".join(x.strip() for x in m.group("body").strip().split("\n") if x.strip())
            if not TIME_RX.search(body) and "closed" not in body.lower():
                continue
            key = (kind, _norm(body))
            if key in seen:
                continue
            seen.add(key)
            found.append((kind, body, c, rank))
    found.sort(key=lambda f: (f[2]["tier"], -int(_pub(f[2]).replace("-", "") or 0), f[3]))
    return found


def answer_hours(question, hits, tokens, idf):
    blocks = hours_blocks(hits, "")
    if not blocks:
        return None
    want = "drive-up" if re.search(r"drive", question, re.I) else "lobby"
    same = re.search(r"\b(same|all|every|differ|vary)\b", question, re.I)
    pick = [b for b in blocks if b[0] == want] or blocks
    place = [w for w in re.findall(r"(?<!^)\b[A-Z][a-z]{3,}\b", question)
             if w.lower() not in GENERIC and w.lower() not in (
                 "lobby", "dupaco", "branch", "drive", "hours", "monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday", "today", "october", "november", "december", "september")]
    if place:  # hours of a named branch must come from text that names it
        pick = [b for b in pick if all(p.lower() in (b[2]["text"] + b[2]["title"]).lower() for p in place)]
        if not pick:
            return REFUSE
    out = pick[:4] if same else pick[:1]
    text = " ".join(f"{k.capitalize()} hours: {body}" for k, body, _, _ in out)
    if same:
        text = "Hours vary by branch. " + text
    else:
        text += " Hours can vary by branch."
    return text, [b[2] for b in out]


def answer_address(question, hits, tokens, idf):
    qtok = set(tokens(question)) - {"address", "branch", "locat", "location", "street", "locate"}
    cands = []
    for rank, c in enumerate(hits):
        for m in ADDR_RX.finditer(c["text"].replace("\n", " ")):
            addr = m.group(0)
            ov = len(qtok & set(tokens(addr)))
            before = c["text"].replace("\n", " ")[max(0, m.start() - 40):m.start()]
            kind = 2 if re.search(r"Branch", before) else (-2 if re.search(r"ATM", before) else 0)
            if ov:
                cands.append((ov * 2 + kind + (3 - c["tier"]) - 0.1 * rank, addr, c))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    return f"{cands[0][1]}.", [cands[0][2]]


_URL = re.compile(r"(?P<anchor>[^\n<>|]{0,80}?)\s*<(?P<url>https?://[^>\s]+)>")
_BAD_HOST = re.compile(r"info\.dupaco\.com|pantheonsite|staging|\.cdn|hubspot|/hubfs|/hs-fs|\.(?:png|jpe?g|svg|pdf)$|"
                       r"facebook|linkedin|instagram|twitter|youtube", re.I)
_ACTION = re.compile(r"\b(join|apply|enrol|open an? |sign up|become a member)\b", re.I)


def answer_link(question, hits, tokens, idf):
    """The page that holds the topic (first official hit), plus an action URL (join./apply) if asked."""
    urls, chunks = [], []
    qtok = set(tokens(question, True))
    if _ACTION.search(question):
        for c in hits[:10]:
            if c["tier"] > 2:
                continue
            for m in _URL.finditer(c["text"]):
                u = m.group("url")
                if re.match(r"https://join\.dupaco\.com/?$", u):
                    urls.append("https://join.dupaco.com/")
                    chunks.append(c)
                    break
            if urls:
                break
    key = _key_tokens(set(tokens(question)), idf, _proper_tokens(question, tokens))
    for c in hits[:8]:
        if c["tier"] > 1 or _BAD_HOST.search(c["url"]):
            continue
        base = c["url"].split("#")[0].rstrip("/")
        if base in urls or re.search(r"/(newsroom|learn|calculators)\b", base):
            continue
        path_tok = set(tokens(re.sub(r"^https?://[^/]+", "", base).replace("/", " ").replace("-", " ")))
        if key and not (key & (path_tok | set(tokens(c["title"] + " " + c["heading"])))):
            continue
        frag = "#rates" if re.search(r"\brates?\b", question, re.I) and "|" in c["text"] and "%" in c["text"] else ""
        urls.append(base + frag)
        chunks.append(c)
        if len(urls) >= 2:
            break
    # explicit anchors on official pages ("find a branch or ATM" -> /contact#locator)
    qk = set(tokens(question)) - {"link", "url", "website", "site", "page", "use", "dupaco", "see"}
    best = None
    for rank, c in enumerate(hits[:12]):
        if c["tier"] > 2:
            continue
        for m in _URL.finditer(c["text"]):
            u = m.group("url")
            if _BAD_HOST.search(u) or "dupaco.com" not in u or re.search(r"/(newsroom|learn|calculators)\b", u):
                continue
            ov = len(qk & set(tokens(m.group("anchor") + " " + re.sub(r"^https?://[^/]+", "", u).replace("-", " ")))) 
            if ov >= 2 and (best is None or ov > best[0]):
                best = (ov, u.split("#")[0].rstrip("/"), c)
    if best and best[1] not in urls:
        urls.append(best[1])
        chunks.append(best[2])
    if not urls:
        return None
    return "You can use " + " or ".join(urls[:3]) + ".", chunks[:3]


def answer_dated(question, hits, tokens, idf, kind):
    """`date` and `count` questions: the sentence carrying a date / number; newest dated source wins."""
    qtok = set(tokens(question, True))
    key = _key_tokens(set(tokens(question)), idf, _proper_tokens(question, tokens))
    rx = DATE_RX if kind == "date" else COUNT_RX
    cands = []
    for sent, c, rank, para in _units(hits):
        if not rx.search(sent):
            continue
        if kind == "count" and key and not (key & set(tokens(sent))):
            continue
        s = _unit_score(sent, c, rank, qtok, idf, tokens, key=key)
        if s:
            cands.append((s, sent, c, para))
    if not cands:
        return None
    if kind == "count":  # same statement in many releases: the most recently published one is current
        newest = max(_pub(c[2]) for c in cands)
        cands = [c for c in cands if _pub(c[2]) == newest] or cands
    cands.sort(key=lambda x: -x[0])
    top = cands[0]
    out, used = [top], {_norm(top[1])}
    for s, sent, c, para in cands[1:6]:
        if len(out) >= 2 or s < 0.6 * top[0] or _norm(sent) in used:
            continue
        out.append((s, sent, c, para))
        used.add(_norm(sent))
    return " ".join(_strip_links(o[1]) for o in out), [o[2] for o in out]


# ── generic ──────────────────────────────────────────────────────────────────

def _with_next(sent: str, para: str) -> str:
    """The sentence plus the one after it ("...the Home$tart program. This initiative provides up to $15,000")."""
    flat = _strip_links(" ".join(x.strip() for x in para.split("\n")))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])", flat)
    s0 = _strip_links(sent).strip(" -|*#")
    for i, x in enumerate(parts):
        if s0[:40] in x:
            nxt = parts[i + 1] if i + 1 < len(parts) else ""
            if nxt and (re.match(r"(This|It|These|They|That|Those|The (program|initiative|offer|account|card))\b", nxt)
                        or re.search(r"[$\d]", nxt)) and len(x) + len(nxt) < 600:
                return f"{x} {nxt}"
            return x
    return s0


def answer_general(question, hits, tokens, idf):
    qtok = set(tokens(question, True))
    key = _key_tokens(set(tokens(question)), idf, _proper_tokens(question, tokens))
    wants_number = bool(re.search(r"\b(how much|amount|cost|fee|much|number)\b", question, re.I))
    cands = []
    for sent, c, rank, para in _units(hits[:20]):
        s = _unit_score(sent, c, rank, qtok, idf, tokens, wants_number, key=key)
        if s and re.search(r"\b(first|second|third|fourth|fifth|sixth)\s+(time|year|consecutive)", sent, re.I):
            s += 1.0  # "the fifth time ..." statements answer recognition / history questions
        if s:
            cands.append((s, sent, c, para))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    top = cands[0]
    chunks = [top[2]]
    # the sentence is shown with its surrounding bullet list / paragraph (facts sit next to each other)
    ptxt = _strip_links(" ".join(x.strip(" -*#") for x in top[3].split("\n") if x.strip()))
    out = [ptxt if len(ptxt) <= 650 else _strip_links(_with_next(top[1], top[3]))]
    used = _norm(" ".join(out))
    for s, sent, c, para in cands[1:]:
        if len(out) >= 3:
            break
        if s < 0.6 * top[0] or _norm(sent) in used:
            continue
        out.append(_with_next(sent, para))
        used += " " + _norm(sent)
        chunks.append(c)
    return " ".join(out), chunks


# ── entry point ──────────────────────────────────────────────────────────────

def _covered(question, c, tokens, idf, thresh=0.45, strict=False):
    """The chunk must hold most of the question's distinctive terms (top 3 by idf, plus named things)."""
    qtok = set(tokens(question))
    ranked = sorted((t for t in qtok if t in idf and t not in CHATTER and not t[0].isdigit()), key=lambda t: -idf[t])[:3]
    key = set(ranked) | (_proper_tokens(question, tokens) & qtok)
    total = sum(idf.get(t, 0) for t in key) or 1.0
    ctok = set(tokens(f"{c['title']} {c['heading']} {c['url']} {c['text']}"))
    if strict and ranked:  # weak row match: every top-distinctive term (e.g. "construction") must be present
        if any(t not in ctok for t in ranked if idf[t] >= 0.85 * idf[ranked[0]]):
            return False
    return sum(idf.get(t, 0) for t in key & ctok) / total >= thresh


_PIDF_CACHE: dict = {}


def answer(question: str, hits: list[dict], tokens: Callable, idf: dict):
    """Return (text, [supporting chunks]) or None to refuse.

    Tokens are compared on their first 6 letters so word variants match
    (recognition / recognized, certificate / certificates, eligible / eligibility)."""
    pidf = _PIDF_CACHE.get(id(idf))
    if pidf is None:
        pidf = {}
        for t, v in idf.items():
            p = t[:6] if len(t) > 6 else t
            pidf[p] = max(pidf.get(p, 0.0), v)
        _PIDF_CACHE[id(idf)] = pidf

    def ptok(text, expand=False):
        return [t[:6] if len(t) > 6 else t for t in tokens(text, expand)]

    return _answer(question, hits, ptok, pidf)


def _answer(question: str, hits: list[dict], tokens: Callable, idf: dict):
    if not hits:
        return None
    res = _answer_inner(question, hits, tokens, idf)
    spec = re.findall(r"\b\d+/\d+\b", question)
    if res and spec and not any(sp in res[0] for sp in spec):
        return None
    if res and _ACTION.search(question) and "join.dupaco.com" not in res[0]:
        for c in hits[:12]:
            if c["tier"] <= 2 and re.search(r"<https://join\.dupaco\.com/?>", c["text"]):
                return res[0] + " Start online at https://join.dupaco.com/.", res[1] + [c]
    return res


def _answer_inner(question: str, hits: list[dict], tokens: Callable, idf: dict):
    if not hits:
        return None
    kind = intent(question)
    if _unknown_entities(question, idf, tokens):
        return None  # asks about a named thing the site never mentions

    # official pages first: archived/old pages only if nothing better
    pool = [h for h in hits if h["tier"] <= 3] or hits

    if kind == "hours":
        r = answer_hours(question, pool, tokens, idf)
        if r is REFUSE:
            return None
        if r:
            return r
    elif kind == "address":
        r = answer_address(question, pool, tokens, idf)
        if r:
            return r
    elif kind == "link":
        r = answer_link(question, pool, tokens, idf)
        if r:
            return r
    elif kind in ("date", "count"):
        r = answer_dated(question, pool, tokens, idf, kind)
        if r and _covered(question, r[1][0], tokens, idf, 0.4):
            return r

    spec = re.findall(r"\b\d+/\d+\b", question)  # "7/1 ARM": a product we may not offer at all
    if spec and not any(sp in h["text"] for sp in spec for h in pool):
        return None
    if RATE_Q.search(question):
        row = best_row(question, pool, tokens)
        specific = row and row[2] >= 6 and row[3] >= 2 and row[4] >= 1
        if specific and re.search(r"\brates\b", question, re.I) and not re.search(r"\d", question):
            tied = near_rows(question, pool, tokens, row)
            if len(tied) > 1:
                return " ".join(t for t, _ in tied), [c for _, c in tied][:3]
        if specific and not (BROAD.search(question) and row[2] < 12):
            text = row[0]
            if "$" in text and not re.search(r"\d[\d,]{3,}", question):
                extra = sibling_rows(row[1], text, tokens)  # no amount named: show the other loan-size tiers too
                text = " ".join([text] + extra)
            return text, [row[1]]
        facts = rate_facts(question, pool, tokens, idf)
        if facts and (BROAD.search(question) or not specific):
            return " ".join(t for _, t, _ in facts), [c for _, _, c in facts][:3]
        if specific:
            return row[0], [row[1]]

    r = answer_general(question, pool, tokens, idf)
    if not r or not _covered(question, r[1][0], tokens, idf):
        return None
    if RATE_Q.search(question) and "%" not in r[0]:
        return None  # a rate question needs a published rate
    return r
