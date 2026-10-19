"""Build a held-out dev set (not part of the submission) to judge designs beyond the 40 public questions.

  A. 2 rewordings of every public question (same gold fields, same group)
  B. new-topic questions grounded on official pages / press releases of the newest crawl,
     with stale values from the oldest crawl's version of the same page as must_not_include
  C. as_of questions on official values that changed between crawls (timeline.csv)
Run: .venv/bin/python <this> ; writes dev_eval.jsonl next to this script.
"""
import csv, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KIT = Path("/Users/felix/projects/bankgpt2/hackathon-knowledge-kit-participant")
OUT = Path(__file__).resolve().parent  # eval/dev/
for line in (KIT / "solution/.env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
import anthropic

ws = os.getenv("ANTHROPIC_WORKSPACE_ID")
client = anthropic.Anthropic(timeout=300, default_headers={"anthropic-workspace-id": ws} if ws else None)
MODEL = "claude-opus-5-5"
usage = [0, 0]


def ask(prompt: str, max_tokens: int = 8000) -> str:
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, thinking={"type": "adaptive"},
                               output_config={"effort": "medium"},
                               messages=[{"role": "user", "content": prompt}])
    usage[0] += r.usage.input_tokens; usage[1] += r.usage.output_tokens
    return "".join(b.text for b in r.content if b.type == "text")


def jarr(s: str) -> list:
    m = re.search(r"\[.*\]", s, re.S)
    return json.loads(m.group(0)) if m else []


def clean(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"\(data:[^)]*\)", "", md)
    return re.sub(r"\n{3,}", "\n\n", md)


def pages(snap: str) -> dict:
    m = json.loads((KIT / "snapshots" / snap / "manifest.json").read_text())
    rows = m["pages"] if isinstance(m, dict) else m
    out = {}
    for p in rows:
        f = KIT / "snapshots" / snap / p["file"] if "file" in p else None
        if f is None:
            f = next(iter((KIT / "snapshots" / snap / "pages").glob(p.get("filename", "") or "_none_")), None)
        if f and f.exists():
            out[p["url"]] = (p["title"], clean(f.read_text()))
    return out


NEW = pages("05_2026-09-28_latest")
OLD = pages("01_2026-05-12_baseline")

# ---------------- A. rewordings of the public questions ----------------
public = [json.loads(l) for l in open(KIT / "eval/public_eval.jsonl")]
PARA = """Here are member questions for a credit-union assistant. For EACH, write 2 rewordings a real member might type: different vocabulary and sentence shape, same meaning and same specifics (same product, term, tier, branch, date). Keep it natural; one may be casual/short. Do not add or drop qualifiers.
Return ONLY a JSON array: [{"id": "...", "rewordings": ["...", "..."]}]

""" + "\n".join(f'{q["id"]}: {q["question"]}' for q in public)


def part_a() -> list:
    out = []
    by = {q["id"]: q for q in public}
    for it in jarr(ask(PARA)):
        q = by.get(it.get("id"))
        if not q:
            continue
        for i, text in enumerate(it.get("rewordings", [])[:2], 1):
            out.append(dict(q, id=f'{q["id"]}-p{i}', question=text, group=q.get("group") or q["id"],
                            origin="paraphrase"))
    return out


# ---------------- B. new topics grounded on pages ----------------
TOPIC_URLS = [
    "/about/join", "/about/member-services", "/business/checking-accounts", "/business/credit-cards",
    "/business/savings-accounts", "/business/cash-management", "/business/loans", "/contact",
    "/membership", "/personal/checking-accounts", "/personal/insurance", "/personal/loans/home-equity",
    "/personal/loans/debt-consolidation", "/personal/loans/loan-protection", "/personal/loans/personal-loans",
    "/personal/loans/recreational-loans", "/personal/loans/student-loans", "/personal/savings-account",
    "/rates", "/business-rates", "/personal/retirement", "/online-and-mobile-banking/mobile-payments",
    "/first-time-homebuyer-grant", "/hardship-grant", "/scholarships", "/shine",
    "/savings-account/online-savings-account", "/financial-services", "/careers", "/membermoneymarket-lp",
    "/business/credit-lines", "/about", "/switch-and-earn", "/personal/credit-cards",
]


def find(pages_: dict, path: str):
    for u, v in pages_.items():
        if u.replace("https://www.dupaco.com", "").rstrip("/") == path.rstrip("/"):
            return u, v
    return None, None


GEN = """You are writing a hidden test set for a credit-union member assistant. It answers from a crawl of dupaco.com taken on September 28, 2026 ("today"). The scorer checks that every `must_include` string appears in the answer (numbers compared by value, text case-insensitively) and gives ZERO if any `must_not_include` string appears.

Below is the CURRENT version of one official page, and (if different) the version of the same page from May 12, 2026.

Write 3 questions a real member would ask that THIS current page answers clearly and unambiguously:
- mix the kinds: a specific value with its tier/term/qualifier (tier_disambiguation), a "list all the options" question (list_completeness), a who/what/how/eligibility question (entity_attribution / current_value), a stale_trap where the May value differs from today's;
- must_include: 1-4 SHORT strings copied exactly from the current page (a number like "3.40%", a name like "Samsung Pay", a phone number); only the values a correct answer cannot omit;
- must_not_include: values that would be WRONG for this question: the May-version value if it changed, or the value of a neighbouring tier/term that a careless reader might quote. Empty list if none. Never list a string that a correct answer could legitimately contain;
- the question must not name the answer; avoid questions whose answer depends on reading a table position-by-position beyond what is stated.
- category one of: tier_disambiguation, list_completeness, entity_attribution, current_value, stale_trap, branch_info.
Return ONLY a JSON array: [{{"question": "", "expected_answer": "", "must_include": [], "must_not_include": [], "category": ""}}]

URL: {url}
TITLE: {title}
=== CURRENT PAGE (2026-09-28) ===
{cur}
=== MAY 12, 2026 VERSION ===
{old}"""


def gen_page(path: str) -> list:
    url, cur = find(NEW, path)
    if not url:
        return []
    _, old = find(OLD, path)
    old_txt = "(identical or not crawled)" if not old or old[1] == cur[1] else old[1][:12000]
    try:
        items = jarr(ask(GEN.format(url=url, title=cur[0], cur=cur[1][:16000], old=old_txt)))
    except Exception as e:
        print("fail", path, e, file=sys.stderr)
        return []
    out = []
    for i, it in enumerate(items, 1):
        if not it.get("question") or not it.get("must_include"):
            continue
        out.append({"id": f"new{path.replace('/', '_')}-{i}", "question": it["question"], "as_of": None,
                    "expected_answer": it.get("expected_answer", ""), "must_include": it["must_include"],
                    "must_not_include": it.get("must_not_include", []), "category": it.get("category", "general"),
                    "weight": 1.0, "expected_sources": [url], "group": None, "origin": "new_topic"})
    return out


# ---------------- C. as_of on changed official values ----------------
SNAPS = [("01_2026-05-12_baseline", "2026-05-12"), ("02_2026-05-19_recrawl", "2026-05-19"),
         ("03_2026-06-09_recrawl", "2026-06-09"), ("04_2026-08-27_rebaseline", "2026-08-27"),
         ("05_2026-09-28_latest", "2026-09-28")]
ASOF = {"2026-05-19": "2026-05-31", "2026-06-09": "2026-07-15", "2026-08-27": "2026-09-10", "2026-05-12": "2026-05-15"}
MON = {"05": "May", "06": "June", "07": "July", "08": "August", "09": "September"}


def part_c() -> list:
    out, seen = [], set()
    rows = list(csv.DictReader(open(KIT / "results/fact_table/timeline.csv")))
    for r in rows:
        if r["changed"] != "yes" or r["kind"] != "APY" or ":" in r["label"]:
            continue  # IRA duplicates of the same rows add little
        vals = [(d, r[s]) for s, d in SNAPS if r[s]]
        for (d, v) in vals[:-1]:
            others = sorted({x for _, x in vals if x != v})
            if not others or (r["label"], v) in seen:
                continue
            seen.add((r["label"], v))
            a = ASOF[d]
            name = r["label"].replace("Fixed ", "").replace("Certificate ", "")
            name = name if "Certificate" in r["label"] or "Special" in r["label"] else name + " certificate"
            out.append({"id": f"asof-{len(out)+1}", "as_of": a,
                        "question": f"What was the APY on the {name} as of {MON[a[5:7]]} {int(a[8:])}, 2026?",
                        "expected_answer": f"{v} APY", "must_include": [v], "must_not_include": others,
                        "category": "temporal_as_of", "weight": 1.5, "expected_sources": [r["official url"]],
                        "group": None, "origin": "as_of"})
    return out


if __name__ == "__main__":
    print("pages", len(NEW), len(OLD), file=sys.stderr)
    with ThreadPoolExecutor(8) as pool:
        fa = pool.submit(part_a)
        fb = list(pool.map(gen_page, TOPIC_URLS))
    items = fa.result() + [x for b in fb for x in b] + part_c()
    with open(OUT / "dev_eval.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    from collections import Counter
    print(len(items), Counter(i["origin"] for i in items), "tokens in/out", usage, file=sys.stderr)
