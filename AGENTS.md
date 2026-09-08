# AGENTS.md — Agent Operating Rules

> Format note: this file is structured for agent parsing. Every rule = one
> atomic, testable statement. No narrative paragraphs. `PLAN.md` = what to
> build. This file = how to build it. Conflict → this file wins.

---

## 1. EXECUTION ORDER

- FOLLOW `PLAN.md` step numbers strictly, in order
- DO NOT start Day N+1 tasks before Day N checkpoint is fully passed
- DO NOT mark a step done without its checkpoint criteria passing
- IF a step is ambiguous → pick smallest testable implementation, do not expand scope

---

## 2. DOMAIN RULES (R1–R7) — APPLY TO EVERY GENERATED REPORT

| ID | Rule |
|---|---|
| R1 | NO HALLUCINATION — finding must exist in `dictation` or `template_content` only |
| R2 | CORRECT ROUTING — finding → matching anatomical field label, never wrong field |
| R3 | PRESERVE UNTOUCHED CONTENT — copy template text character-for-character if not addressed by dictation, at both field level AND clause level (a touched field may still contain an untouched clause, e.g. "No X or Y." with only X dictated — Y's clause must survive verbatim) |
| R4 | PRESERVE LABELS + ORDER — output field-label set and order must exactly match template |
| R5 | NEGATION/LATERALITY/MEASUREMENTS/SEVERITY/ACUITY ARE HIGHEST PRIORITY — always explicit: side, presence/absence, severity, acuity, measurement+unit |
| R6 | IMPRESSION = SUMMARY ONLY — no new claims beyond what changed FINDINGS support |
| R7 | FIXED OUTPUT STRUCTURE — exactly one `FINDINGS:` + one `IMPRESSION:` section, nothing else |

- EXCEPTION to R4: `OTHER FINDINGS:` may receive routed content only if it already exists in template
- LABEL MATCHING: normalize (strip colon/whitespace, uppercase) before comparing;
  hierarchical child labels (e.g. `Medial meniscus:` under `Menisci:`) count as fields
- NORMAL FAST PATH: dictation `normal` (case-insensitive) → copy template with zero edits
- FEW-SHOT ROUTING: key examples by modality + label set, never by body_part alone
  (test contains unseen body_part values)
- VIOLATION of R1–R7 → reject the change, even if it "reads better"
- R8 | DATA DISCLAIMER — README.md must state: "This benchmark uses de-identified data and must not
  be used for clinical decisions." (per competition data notice)

---

## 3. PROMPT RULES

### General
- OUTPUT INSTRUCTION: always "return ONLY [format]. No preamble, no markdown fences, no explanation."
- RESTATE relevant R1–R7 rules inside the prompt itself (model doesn't read this file)
- FEW-SHOT EXAMPLES: pull from real `train.csv` rows only, never invented
- FEW-SHOT COVERAGE: minimum 3 different modalities

### API volume
- ESTIMATE: ~770 rows (train+test) × 2 LLM calls (extraction + impression) + retries ≈ ~1,600 calls
- USE: simple sequential or lightly-batched calls — no async complexity needed at this volume
- IF rate-limited: add basic exponential backoff in `extract.py` / `impression.py`

### Stage 1 — Extraction prompt
- OUTPUT FORMAT: strict JSON array only, no free text outside JSON
- SCHEMA (clause-level, not whole-field — required for R3 compliance on compound sentences):
  ```json
  [{"field_label": "PLEURA",
    "matched_template_span": "No pleural effusion or pneumothorax.",
    "unaffected_subclause": "No pneumothorax.",
    "new_clause_text": "Small right pleural effusion.",
    "laterality": "right", "severity": "small", "acuity": null,
    "polarity": "present", "measurement": null, "measurement_unit": null}]
  ```
- `matched_template_span` / `unaffected_subclause` MUST be verbatim quotes from `template_content`
  — instruct the model to copy, never paraphrase, these two fields
- `field_label` MUST match a template field label after normalization (strip colon/whitespace,
  uppercase) — this normalization is Stage 2's job, not the prompt's, but the prompt should tell
  the model to copy the template's label text as-is

### Stage 3 — Impression prompt
- INPUT: only findings marked "changed" — NEVER raw dictation, NEVER full merged report
- REASON: prevents re-summarizing untouched/normal fields

### Determinism (applies from first implementation, not just final notebook)
- ALL Stage 1 and Stage 3 API calls use `temperature=0` — from first implementation onward
- REASON: val-loop stability (Checkpoint 3 requires two stable consecutive runs) is impossible
  otherwise; do not defer this to Day 4 notebook-export only

### Stage 2 — HARD RULE
- NEVER implement merging/preservation via LLM call
- Stage 2 = pure Python function, always
- This is the single most important architecture rule in this repo
- Stage 2 MUST verify `matched_template_span` is an exact substring of the template field before
  applying an edit; on failure, keep the whole field unchanged and log — never guess a splice
  location

---

## 4. CODING STANDARDS

| Rule | Detail |
|---|---|
| Language | Python only — no JS/TS, no notebook-as-primary-code |
| Structure | plain functions, no classes unless state truly required |
| Dependencies | no LangChain/orchestration frameworks — direct API calls only |
| File responsibility | one file = one stage (per PLAN.md repo structure) |
| Docstrings | every function: one line, input → output |
| Type hints | required on all function signatures |
| Exceptions | no bare `except:` — catch specific exceptions, log `case_id` on failure |
| State | no global mutable state — pass data explicitly between stages |
| Tests | every pure-logic module (`merge.py`, `validator.py`, `scorer.py`) needs a test file before "done" |
| Line length | ≤ 100 chars |
| Dead code | no commented-out code in commits |

---

## 5. SECRETS

- API keys: default provider is local Ollama (`LLM_PROVIDER=ollama`) — no key needed
- IF `LLM_PROVIDER=anthropic`: key via `os.environ["ANTHROPIC_API_KEY"]` ONLY
- NEVER hardcode a key — not in code, notebooks, prompts, tests, comments
- `.env` → in `.gitignore`, never committed
- `.env.example` → committed, placeholder value only
- BEFORE every commit: grep diff for `sk-`, `api_key`, `ANTHROPIC_API_KEY=<real value>` → if found, STOP, remove, then commit

---

## 6. VALIDATION GATES (must pass before advancing)

| Gate | Before | Requirement |
|---|---|---|
| G1 | Stage 2 marked done | `test_merge.py` covers: untouched-preserved, touched-replaced, compound-sentence clause split (touched clause replaced, sibling clause preserved verbatim), multi-finding-in-one-field ordering, `matched_template_span` verification failure → safe fallback, unmapped-routed-to-OTHER, unmapped-dropped-with-log |
| G2 | Stage 4 marked done | `test_validator.py` covers: missing field, extra field, wrong order, malformed output, untouched-field byte-mismatch (R3 violation) — all correctly flagged |
| G3 | Day 3 starts | `scorer.py` reproduces plausible score on chest X-ray worked example |
| G4 | Day 4 test.csv run | mean val RES computed 2+ times with documented improvement, OR explicit README note why not |
| G5 | submission upload | every row in submission.csv passes Stage 4 validator |

---

## 7. GIT RULES

### Branching
- `main` = always working state
- feature branches: `day{N}/{short-task}` (e.g. `day1/res-scorer`, `day2/extraction-stage`)
- merge to `main` ONLY after that step's PLAN.md checkpoint passes

### Commits
- ONE logical change per commit = one PLAN.md sub-step
- NO end-of-day dump commits
- FORMAT:
  ```
  <stage>: <short imperative description>

  Refs: PLAN.md step <step number>
  ```
  Example: `merge: preserve untouched template fields exactly` / `Refs: PLAN.md step 2.2.2`

### Never commit
- `.env` / API keys / any secret
- intermediate/debug `outputs/submission.csv` (only final commit)
- raw LLM response dumps (log locally, don't commit)
- `__pycache__/`, `.ipynb_checkpoints/`

### `.gitignore` (minimum contents)
```
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
outputs/*.csv
!outputs/.gitkeep
```

### Other
- TAG final submission commit: `git tag submission-v1`
- NO force-push to `main`

---

## 8. ERROR HANDLING

| Situation | Action |
|---|---|
| Stage 1 returns malformed JSON | retry once with parse error appended to prompt |
| Stage 1 fails twice | log `case_id`, fall back to `template_content` copied unchanged as the report (valid FINDINGS/IMPRESSION structure, passes Stage 4, worst-case RES for that row) — every test.csv case_id must appear exactly once in submission.csv, so an undefined placeholder is not a legal fallback. THIS RULE IS AUTHORITATIVE — PLAN.md orchestration/submission steps must implement it exactly; "hard-fail with no row produced" is NOT a legal deviation, since it would leave a case_id missing from submission.csv |
| Stage 4 validator fails | auto-retry Stage 1 with failure reason, OR flag for manual review |
| Local RES disagrees with manual read | STOP — re-check scorer implementation before trusting further val results |
| Before writing submission.csv | assert `len(output_case_ids) == len(test_case_ids) == 132` (and no duplicates) — enforces the fallback rule above is actually working, not just documented |

---

## 9. AUTONOMY BOUNDARY

### May proceed WITHOUT asking
- write/edit code in `src/`, `tests/`, `prompts/`, `scripts/`
- run pipeline on train/val data
- commit to a feature branch

### MUST STOP and ask user first
- merge feature branch → `main`
- run pipeline on real `test.csv`
- generate + upload `submission.csv`
- share Kaggle notebook / submit to Kaggle
