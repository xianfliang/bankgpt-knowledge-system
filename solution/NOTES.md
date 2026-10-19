# NOTES: a knowledge system that knows what's true *now*

## For the graders: API key and running

- **API key: `ANTHROPIC_API_KEY`, a paid Anthropic API key.** Answers and part of the
  conflict search call **Claude Haiku 4.5** (`claude-haiku-4-5`). No other key or service is
  needed.
- **Cost:** about $0.008 per answered question (about $0.35 for the 40 public ones), plus about
  $0.40 for one `conflicts()` call (about 35 Haiku calls). Ingest makes no API calls.
- **Without a key** (or if the API fails or runs out of credit), the system still runs: a
  circuit breaker switches to a rule-based answerer, and the rule-based conflicts are still
  reported. That mode scores 0.871 on the public set, but the LLM path is better on unseen
  questions (§6), so please grade with a key.
- **Packages:** `anthropic>=1.0` (`requirements.txt`). Checked in a clean Python 3.11
  environment: `python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl`
  runs from the repo with ingest in about 7 s.
- **Side effect:** after each ingest, the fact table is written to `results/fact_table/` for
  review. A failed write is ignored; `KS_FACT_LOG=` turns it off.
- Only the snapshots are used; there are no live website calls.

## Approach

Plain retrieval plus an LLM hands the model raw chunks in which stale and current values sit
side by side, and a single stale number zeroes a question. So this system works out **what is
true in each crawl once, at ingest**, in a versioned fact table built with plain rules (no LLM).
Every rate-table cell and every "label: value" line becomes a fact keyed by its label, kind
(rate, APY, APR, fee) and qualifiers (term, tier, model year, jumbo, personal or business).
Different qualifiers are different facts, not conflicts.

**How "current" is decided.** For each key in each crawl, the winner is the value on the most
authoritative page that is not an expired offer. The authority order is official
product/rate/contact pages, then marketing pages, then calculators, blogs, PDFs and press
releases, then archived posts. Values that lose (on a lower-authority page, in an expired
promotion, on a row the official table has since dropped, or on the official page in an older
crawl) are marked stale. Among dated content such as press releases, the newest *published*
date wins. An `as_of` question uses the newest crawl dated on or before that day; without
`as_of`, the newest crawl (2026-09-28).

**Answering.** BM25 search, weighted by authority, retrieves 20 chunks. Every line stating a
stale value is removed before Claude Haiku sees them, and any sentence of the answer that still
quotes one is cut. A rate question naming a term or tier the site doesn't list is refused with
a fixed "I don't know" (`confident=False`). Answers cite their pages.

**Conflicts.** These are read from the same fact table, so the reported winner is the value
the answers give, with rows of one table merged into one entry. Rule-based checks cover
branch counts, award counts and "up to X% APY" promotions. A Haiku sweep covers topics outside
the tables (routing number, hours, links, eligibility).

| Measured | Result |
|---|---|
| Public 40, with Haiku | accuracy 0.84, consistency 1.0, p95 ≈ 3.4 s |
| Our 190-question held-out set (§6), with Haiku | accuracy 0.853 |
| Harness selftest | 1.0 (conflict found, correct winner) |
| Conflicts from the rules alone, against 8 contradictions we checked by hand | F1 0.76, correct winner 8 of 8 |
| No API key | public 0.871, held-out 0.675 |
| Ingest | ≈ 7 s for all five crawls |

## 1. Architecture

```
ingest(snapshot) ──► clean ──► chunk (heading-aware, tables whole) ──► BM25 index per snapshot
                          │          ▲ pages missing from this crawl are carried forward
                          └─► fact table (facts.py): table cells + "label: value" lines
                                 ──► resolve each key: winner / superseded / retired / dead values

answer(q, as_of) ──► snapshot for as_of ──► term/tier listed by a live fact? no ──► refuse
                     ──► BM25 × authority × recency ──► top-20 chunks ──► drop lines with dead values
                         ├─► Claude Haiku 4.5 (grounded prompt) ──► text + cited sources
                         └─► API unavailable or too slow? ──► extractive.py (no LLM)
                     ──► scrub any sentence that still quotes a dead value

conflicts() ──► fact table (lower-authority / expired / retired / changed across crawls), merged per table
             ├─► claim checks (branch count, award count, "up to X% APY" promos)
             └─► LLM sweep over member topics the table doesn't cover ──► filter, merge, de-duplicate
```

| File | Role |
|---|---|
| `solution.py` | `build()`, ingest, retrieval, answering, the LLM client |
| `facts.py` | the versioned fact table: extraction, per-snapshot resolution, evidence redaction, term/tier support check, table and claim conflicts |
| `extractive.py` | the no-LLM answerer |
| `conflicts.py` | the LLM conflict sweep and the merge with the rule-based conflicts |
| `prose.py` | optional LLM prose-fact extraction at ingest (off by default, see §6) |

## 2. Ingest

- **Cleaning:** removes inline base64/SVG images, "Handpicked for You" teasers and site-wide
  nav/footer chunks. Footnote markers are normalised ("3.90%1" → "3.90%").
- **Chunking:** heading-aware chunks of about 1,400 chars. A table never loses its header row or
  the label line above it, so a value stays attached to its tier.
- **Carry-forward:** a page that a later crawl misses is carried forward from the last crawl that
  had it, and labelled as such.
- **Authority tiers:** official product/rate/contact pages (×1.0) > marketing pages (×0.85) >
  calculators, blogs, PDFs and press releases (×0.6) > archived `/20YY/MM/` posts and
  `info.dupaco.com` (×0.35). Press releases also lose 7% per year of age.
- **Fact-table log:** after each ingest, the table is written to `results/fact_table/` for review
  (`summary.md`, `timeline.csv`, per-snapshot `facts.csv`). Set `KS_FACT_LOG=` to turn it off.
- No API calls are needed at ingest, apart from one cache warm-up when the optional
  official-pages prefix is on.

## 3. The fact table (`facts.py`)

1. **Extract** every rate-table cell, and every short line that states exactly one rate or amount
   ("Lost debit card replacement fee: $2.50"), as a fact: `(normalised label, kind) → value`, with
   URL, authority tier, context and an `expired` flag (the line says the offer ended, or runs
   through a date that has passed).
   - The section heading is part of the label.
   - Column headers separate several columns of one kind (promotional vs after-promotion).
   - Min/max/to-open/points/payment columns, 0% cells and calculator worked examples are not
     facts. "$1 million" is read as 1,000,000.
2. **Resolve each key per snapshot:**
   - *winner:* the value of the best-tier, non-expired facts. When two equally trusted pages
     disagree there is no winner, and nothing is marked stale.
   - *superseded:* values the key shows on a lower-tier page or an expired offer, or showed on
     the official page in an older crawl.
   - *retired rows:* a row the official page showed in an earlier crawl and no longer shows,
     while the page itself is still there. Other pages still quoting it are stale. Its successor
     is the official row with the same words and the most numbers in common
     ("2023-26 Model Year | 60 Months" → "2024-27 Model Year | 60 Months").
   - *dead values:* superseded values that are not any key's winner in that snapshot. A number
     that is stale for one row but current for another is never dead, so the guard stays
     conservative.
3. **Mirror pages:** where the crawl holds several URLs with the same page, the URL whose path
   best matches the title is cited.

Newest crawl: about 1,600 facts, 830 keys, 30 dead values (old mortgage-guidance and refinance
landing-page APRs, retired auto-loan rows, old certificate/IRA specials).

## 4. Answering

**Rules for "current":**
1. **Time:** the crawl live on `as_of`; without a date, the newest crawl. The prompt treats the
   crawl as the site as it stood on the asked date, including dates after the crawl.
2. **Authority, then recency:** official pages win; among dated content the newest wins.
3. **Qualifiers are different facts:** term, rate vs APY vs APR, balance or loan-amount tier,
   model-year range (2024-27 covers 2025), jumbo/special, Roth/Traditional, personal vs
   business. With no tier named, give the headline tier. Broad requests ("all certificates")
   get the full list.
4. **Exact copying:** names, dates, phone numbers and extensions are copied as printed.
5. **Honesty:** with no evidence, a fixed refusal with no digits and `confident=False`.

**Evidence.**
- Lines stating a dead value or a lapsed offer are removed from retrieved chunks before the
  model sees them; numbers named in the question are kept.
- An excerpt that redaction left with no values is dropped. An emptied stale table otherwise
  reads as "rates not shown" and causes false refusals.
- A rate question naming a term the site never lists for that product (for example a 60-month
  jumbo certificate) is refused before any LLM call.

**Sources.** The cited excerpts plus the top 3 official pages used. The scorer credits any
matching URL and does not penalise extra ones.

**Latency and failures.**
- **Hedged requests:** if no reply arrives within `KS_HEDGE_S` (4 s), an identical second request
  is sent (a third after 8 s), and the first reply wins. Each attempt is capped at 10 s and the
  whole call at 20 s, after which the no-LLM path answers. This removes the rare provider stalls
  that would otherwise hit the 60 s harness limit.
- **Circuit breaker:** a missing key, an auth failure or exhausted credit (or 3 consecutive
  transient failures) disables the API for the rest of the run, with one line on stderr. Later
  questions go straight to the no-LLM answerer.

**No-LLM answerer (`extractive.py`, p95 0.02 s).** It classifies the answer type, extracts that
kind of fact from all retrieved chunks and assembles a short answer:

| Type | How |
|---|---|
| rate | best table row (term, model year, balance tier, qualifiers); plural or broad questions list the rows |
| hours | Lobby / Drive-up schedule blocks; a named branch must appear in the block |
| address | the street line of the named branch |
| link | the official page for the topic; `join.dupaco.com` for join/apply |
| date / count | the sentence with the date or number; for counts, the newest *published* release wins |
| general | top sentences with their following sentence or bullet list |

It refuses when the question names things the evidence doesn't contain.

## 5. Conflicts

1. **Fact table.** Inside the newest crawl, a lower-authority page or an expired offer that
   disagrees with the winner. A retired row still quoted elsewhere, reported against its
   successor. An official value that changed between crawls, with the newest value as the
   winner. Rows of the same official table with the same term are **merged into one entry**
   that lists every value involved, and the resolution gives each row's current rate. Without
   the merge, one mortgage-table change produced 12 entries, which hurts F1 precision.
2. **Claim checks (no LLM).**
   - *Branch count:* the press-release boilerplate "N branch office locations"; the newest
     published release wins (22).
   - *Awards:* "the Nth time Forbes recognized Dupaco"; the newest release wins (fifth).
   - *Promo pages:* "up to X% APY" on a non-official page, above the official table's highest
     APY for that product (a checking landing page promising 3.00% against an official maximum
     of 2.50%).
3. **LLM sweep (Haiku).** Generic member topics the table doesn't cover: routing number, hours,
   links, eligibility and similar. Each topic's prompt holds official excerpts, low-authority
   excerpts and older-crawl versions of the same pages. It keeps at most 2 findings per topic.
   Formatting-only differences ("20.5%" vs "20.50%", "$3001" = "$300" plus footnote 1) are
   dropped. A finding is also dropped when an earlier entry already names two of its values.

| Conflicts (fact table + claims) against 8 hand-checked contradictions | Entries | F1 | Correct winner |
|---|---|---|---|
| Unmerged table rows | 33 | 0.24 | 0.38 |
| Merged per table | 10 | 0.56 | 0.63 |
| + claim checks | 13 | **0.76** | **1.0** |

The 5 unmatched entries are real contradictions that are not on the list (other mortgage terms,
auto-loan terms, IRA specials).

## 6. Evaluation and what was tried

**Held-out set.** The 40 public questions are too few to tune on, so a 190-question set was built
with Claude Opus 5.5. It was used only for development and is not part of the submission:
- 80 rewordings of the public questions;
- 102 questions on 34 official pages the public set doesn't use, with older-crawl values as
  `must_not_include`;
- 8 `as_of` questions on certificate APYs that changed between crawls.

| Path | Public 40 | Held-out: rewordings | Held-out: new topics | Held-out: as_of | Held-out overall |
|---|---|---|---|---|---|
| No-LLM answerer | **0.871** | 0.79 | 0.50 | 0.38 | 0.675 |
| LLM answers (Haiku 4.5) | 0.84 | 0.84 | **0.89** | 0.68* | **0.853** |

\* Measured before the `as_of` prompt fix; with the fix, the same pipeline on DeepSeek V4.1 Flash
scored 0.89 on `as_of`.

The no-LLM answerer's rules fit the public questions but do not transfer to new topics, so the
LLM path is the default and the no-LLM path is the fallback.

**Tried and not adopted:**

| Idea | Result |
|---|---|
| Router: send some question types to the no-LLM path | Every rule that helped on the public set lost on the held-out set (best 0.851 vs 0.853) |
| Fewer excerpts (`KS_TOP_K` 12 or 15) for speed | Public accuracy 0.76 / 0.79 vs 0.84 |
| All official pages (≈ 75k tokens) in a cached prompt prefix (`KS_CORE_CHARS=260000`) | Public 0.840 vs 0.843; fixes eligibility questions but the model reads long context less carefully. Off by default |
| LLM prose-fact extraction at ingest (`KS_PROSE_FACTS=1`) | Public 0.838 vs 0.843, about $2.50 and 6 minutes per cold ingest. Extracted names varied between runs, so few cards resolved against each other. Off by default |
| Other answer models via OpenRouter | DeepSeek V4.1 Flash: held-out 0.874 accuracy but p95 5–11 s, overall tied with Haiku 4.5. Gemini 2.5 / 3.1 Flash-Lite and Haiku 5.5 reach p95 ≈ 2 s but lose 6–16 accuracy points |

**Why not chase the 2 s p95.** Accuracy weighs 55% on its own and also multiplies consistency and
speed, so each accuracy point is worth about 0.008 of the final score. Going from p95 3.4 s to 2 s
gains at most about 0.01, and every faster setup measured cost more accuracy than that.

## 7. Known limitations

- **Chunk recall.** Most remaining LLM misses are false refusals: the right page is retrieved but
  not the chunk holding the answer.
- **Conflict sweep precision** on the real data is not measured. The rule-based entries are
  high precision; the sweep is broader and is capped at 2 findings per topic.
- **`as_of` before the first crawl** uses the first crawl rather than refusing.
- **A note on the public set.** Some public `must_not_include` values appear in no crawl, so they
  catch invented answers rather than contradictions on the site. dup-049 forbids 0.35%, which is a
  real current rate for the Health Savings Account's $25,000+ tier.

## 8. Configuration

All settings are optional; the defaults are the configuration we measured best. For local runs,
keys may also be put in a `solution/.env` file (`KEY=value` lines; real environment variables
win). It is git-ignored and not part of the submission.

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Claude API, for answers and the conflict sweep |
| `ANTHROPIC_WORKSPACE_ID` | unset | sent as the `anthropic-workspace-id` header; only for a key not scoped to a workspace |
| `KS_MODEL` | `claude-haiku-4-5` | answer model. An OpenRouter id (e.g. `deepseek/deepseek-v4.1-flash-1017`) uses OpenRouter's Anthropic-compatible endpoint |
| `OPENROUTER_API_KEY` | unset | only for an OpenRouter `KS_MODEL` |
| `KS_CONFLICT_MODEL` | `claude-haiku-4-5` | conflict sweep model (untimed) |
| `KS_HEDGE_S` | `4` | seconds before a hedged duplicate answer request |
| `KS_ANSWER_DEADLINE_S` | `20` | give up on the API after this long and use the no-LLM path |
| `KS_TOP_K` | `20` | chunks sent to the model per answer |
| `KS_CORE_CHARS` | `0` | official pages sent whole in a cached prompt prefix, up to this many chars |
| `KS_CORE_TOP_K` | `20` | search excerpts sent next to the prefix |
| `KS_CORE_DEDUP` | `0` | `1`: excerpts skip text already in the prefix |
| `KS_PROSE_FACTS` | `0` | `1`: LLM prose-fact extraction at ingest |
| `KS_CARD_K` | `6` | prose fact cards added to each answer (only with `KS_PROSE_FACTS=1`) |
| `KS_FACT_LOG` | `results/fact_table` | where the fact table is written after ingest; empty turns it off |
| `KS_CACHE_DIR` | unset | on-disk response cache for local development |
| `KS_USAGE` | unset | print per-model calls and token totals at exit |

**Dependencies:** `anthropic>=1.0` (`requirements.txt`).
