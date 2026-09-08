# PLAN.md — Radiology Reporting Harness (Natoe.ai)

> Format note: this file is structured for agent parsing. Each step = one atomic,
> checkable action. No narrative text. Execute top to bottom, in order.

---

## SECTION 0 — TASK DEFINITION

- INPUT: `template_content` (normal report) + `dictation` (doctor notes: 1-word
  `normal` OR long detailed up to ~418 words) + context cols (modality, body_part,
  study_description, patient_age_band, patient_sex — context only, never findings)
- OUTPUT: edited report = `FINDINGS:` + `IMPRESSION:` only
- RULE: change ONLY fields dictation addresses
- RULE: copy all other fields from template unchanged
- RULE: field labels + order must match template (match case-insensitively after
  strip-colon/whitespace/uppercase; emit template's original label text).
  Hierarchical labels exist (e.g. `Menisci:` parent + `Medial meniscus:` child,
  183 unique labels) — treat each labeled line as its own field.
- DATA FACTS (measured 2026-09-08): train 636x9, test 132x8; XRAY 380/MRI 158/
  CT 71/USG 27 train; `OTHER FINDINGS:` in 56/636 train + 28/132 test; test has
  unseen body_part (Arm, Heel, Thumb, Bladder, Paranasal sinus, Orbit, Sternum,
  Knee patella, Ankle calcaneous) → route by modality + label set, never by
  body_part alone; dictation `normal` → copy template verbatim (blank lines /
  uppercased labels / numbered impression are free formatting, scorer strips them)
- METRIC: RES (Radiology Edit Score) — lower = better
- METRIC PENALIZES: wrong field routing, hallucinated content, unnecessary rewrites of untouched fields

---

## SECTION 1 — ARCHITECTURE (5 stages)

| Stage | Name | Type | Input | Output |
|---|---|---|---|---|
| 0 | Data layer | code | train.csv/test.csv | validated dataframe + train/val split |
| 1 | Extraction | LLM | dictation | structured findings JSON |
| 2 | Merge | code (no LLM) | template + findings JSON | draft FINDINGS text |
| 3 | Impression | LLM | changed findings only | IMPRESSION text |
| 4 | Validator | code | assembled report | pass/fail + reason |
| 5 | Local scorer | code | generated + reference report | RES score |

- KEY RULE: Stage 2 = pure Python, never LLM. This guarantees untouched fields stay untouched.
- KEY RULE: Stage 5 runs on train/val BEFORE any test.csv generation.

---

## SECTION 2 — REPO STRUCTURE

```
radiology-harness/
├── PLAN.md
├── AGENTS.md
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/
│   ├── data_loader.py      # Stage 0
│   ├── extract.py          # Stage 1
│   ├── merge.py             # Stage 2
│   ├── impression.py        # Stage 3
│   ├── validator.py         # Stage 4
│   ├── scorer.py             # Stage 5
│   └── pipeline.py           # orchestrator
├── prompts/
│   ├── extraction_system_prompt.txt
│   └── impression_system_prompt.txt
├── tests/
│   ├── test_merge.py
│   ├── test_validator.py
│   └── test_scorer.py
├── scripts/
│   ├── run_val.py
│   └── run_submission.py
├── notebooks/
│   └── pipeline_notebook.ipynb
└── outputs/
    └── submission.csv
```

---

## SECTION 3 — DAY 1: UNDERSTAND + BUILD SCORER

### Step 1.1 — Data exploration
- 1.1.1 Load train.csv
- 1.1.2 List all unique field labels across all templates
- 1.1.3 Group by (modality, body_part) → note field differences
- 1.1.4 Manually read 20 random rows: dictation → template → reference report
- 1.1.5 Record findings in README.md under "Data notes"

### Step 1.2 — Build `src/scorer.py`
- 1.2.1 Write text normalizer: lowercase, unicode-normalize, strip punctuation
  (keep signed numbers), remove list markers, split letter-number boundaries,
  standardize units (mm/cm)
  - 1.2.1.1 REQUIRED OPERATION ORDER (steps interact — wrong order breaks signed numbers /
    hyphen handling):
    1. Unicode normalize + lowercase
    2. Protect signed numbers via regex (e.g. `-2` → placeholder token) before any
       punctuation/hyphen stripping touches them
    3. Remove hyphens between letters only (`air-space` → `airspace`; must not match the
       protected-number placeholders)
    4. Strip remaining punctuation
    5. Restore protected signed numbers
    6. Remove leading list markers (`1.`, `2)`, `-`, `*`, `•`)
    7. Split letter-number boundaries into tokens
    8. Standardize unit spellings (mm/cm)
  - 1.2.1.2 Unit test: input containing both a hyphenated word and a signed measurement in the
    same string (e.g. `"-2 mm air-space opacity"`) must normalize with the minus sign intact
    and the hyphen removed
- 1.2.2 Write token weight classifier:
  - 4.0 = negation, laterality, severity, acuity, numbers, units
  - 2.0 = other content words (anatomy, descriptive terms)
  - 0.25 = function words (the, and, of, with)
  - 1.2.2.1 Build explicit word lists per tier as constants inside `src/scorer.py` (not left to
    the LLM/heuristic at score time — no new file, keep it with the function that uses it) — e.g. negation: {no, not, without, absent, negative for, none}; laterality:
    {right, left, bilateral, unilateral}; severity: {trace, tiny, small, mild, moderate, large,
    severe, extensive}; acuity: {acute, subacute, chronic, stable, interval, new, resolved,
    worsening, improving}; numbers/units matched by regex, not a fixed list
  - 1.2.2.2 Any token not matched by a list/regex defaults to the 2.0 "other content word" tier
    unless it is a stopword on the 0.25 function-word list
- 1.2.3 Write weighted word-level Levenshtein distance function
  - 1.2.3.1 Cost = minimum weighted sum of insert/delete/substitute operations aligning submitted
    tokens to reference tokens (ordered — no reordering credit)
  - 1.2.3.2 REQUIRED normalization: divide the raw edit cost by
    `max(total_token_weight(reference), total_token_weight(submitted))`, then cap the result at 1.0
    — this exact normalization is specified in the competition doc and must match it, or local
    val RES will diverge from the real leaderboard score
- 1.2.4 Write field-aware FINDINGS score `F`:
  - field weight = 3 if field changed vs template, 1 if unchanged (determined by diffing
    `template_content` against the reference `report`, train/val only — never used at
    generation/inference time since the reference is unknown then)
  - missing field = score vs empty string
  - unexpected field (present in output, not present in template/reference) → additional penalty:
    treat as extra content scored against an empty reference string; add its edit cost to the
    numerator AND add its field_weight (weight 3, treated as "changed") to the denominator
    `sum(field_weight)` — this prevents a hallucinated extra field from ever *lowering* the
    average cost by inflating the denominator alone
    - unit test: a submission with one correct field + one hallucinated extra field must score
      strictly worse than the same submission without the extra field
  - unsupported unlabelled FINDINGS content (text outside any recognized field label) → same
    empty-reference penalty treatment (numerator + denominator, as above)
  - `F = sum(field_weight × field_word_edit) / sum(field_weight)`
- 1.2.5 Write IMPRESSION score `I` = weighted word edit over full section
- 1.2.6 Combine: `RES_case = 0.65*F + 0.35*I`
- 1.2.7 Write mean-RES function over a list of cases

### Step 1.3 — Test the scorer
- 1.3.1 Unit test: identical text → RES ≈ 0.0
- 1.3.2 Unit test: completely different text → RES near 1.0
- 1.3.3 Unit test: correct content in wrong field label → score worse than same content in right label
- 1.3.4 Sanity test: run on the chest X-ray worked example from competition doc

### Step 1.4 — Train/val split
- 1.4.1 Split train.csv ~80/20
- 1.4.2 Save split indices to file for reproducibility

**CHECKPOINT 1 (must pass before Day 2):**
- [ ] scorer.py passes all unit tests
- [ ] scorer.py implements the exact normalization rule (divide by max total token weight of
  reference/submitted, cap at 1) — not an approximation
- [ ] scorer.py produces plausible score on worked example
- [ ] train/val split saved and reproducible

---

## SECTION 4 — DAY 2: BUILD PIPELINE

### Step 2.0 — API volume estimate
- 2.0.1 Estimate: ~770 rows (train+test) × 2 LLM calls (extraction + impression) + retries
  ≈ ~1,600 calls. `normal` dictations skip both calls via fast path (2.5.0).
- 2.0.2 Use simple sequential or lightly-batched calls — no async complexity needed at this volume
- 2.0.3 If rate-limited: add basic exponential backoff in `extract.py` / `impression.py`
- 2.0.4 Cache every LLM request/response in `outputs/cache.jsonl` keyed by row-hash —
  Day 3 re-runs must not re-pay or re-roll

### Step 2.1 — Stage 1: Extraction prompt
- 2.1.1 Write strict system prompt (see AGENTS.md §3 for rules)
- 2.1.2 Add 3+ few-shot examples from train.csv, covering 3+ modalities
- 2.1.3 Define output schema: JSON array of findings, clause-level (not whole-field text) so
  Stage 2 can preserve untouched portions of a compound template sentence exactly
  ```json
  [{"field_label": "PLEURA",
    "matched_template_span": "No pleural effusion or pneumothorax.",
    "unaffected_subclause": "No pneumothorax.",
    "new_clause_text": "Small right pleural effusion.",
    "laterality": "right", "severity": "small", "acuity": null,
    "polarity": "present", "measurement": null, "measurement_unit": null}]
  ```
  - 2.1.3.1 `matched_template_span` and `unaffected_subclause` MUST be exact literal substrings
    of `template_content` — instruct the model explicitly to copy, not paraphrase, these two
    fields; Stage 2 will reject any entry where this doesn't hold (see 2.2.3.1)
  - 2.1.3.2 `field_label` MUST be copied verbatim from the template's own label text
- 2.1.4 Implement `src/extract.py::extract_findings(row) -> list[dict]`
  - 2.1.4.1 `temperature=0` on this API call — REQUIRED from this first implementation onward
    (not deferred to Day 4 notebook export), so val-loop runs are reproducible (see Checkpoint 3)
- 2.1.5 Add retry-on-malformed-JSON logic (max 1 retry)

### Step 2.2 — Stage 2: Merge (pure code)
- 2.2.1 Parse `template_content` → `{field_label: field_text}` ordered dict
  - 2.2.1.1 Normalize both extraction `field_label` values and template keys before matching:
    strip trailing colon, strip whitespace, uppercase. Match on normalized form; store/emit the
    original template label text (so output labels stay byte-identical to the template)
- 2.2.2 For each field: no matching finding → keep template text unchanged
- 2.2.3 For each field: matching finding(s) → replace/splice text at the clause level:
  - 2.2.3.1 Verify `matched_template_span` is an exact substring of the field's current text.
    If verification fails → do NOT edit this entry; keep the whole field unchanged, log a
    warning with case_id (never guess a splice location)
  - 2.2.3.2 If verified, replace `matched_template_span` with `new_clause_text` in place;
    `unaffected_subclause` needs no separate handling — it is already inside
    `matched_template_span` and survives automatically once only the touched portion is swapped
  - 2.2.3.3 Multi-finding-per-field ordering: apply matched entries in the order their
    `matched_template_span` occurs in the original template field text (left to right), not
    extraction-output order, so field-internal sentence order is preserved
- 2.2.4 Unmapped finding + `OTHER FINDINGS:` exists in template → route there
- 2.2.5 Unmapped finding + no `OTHER FINDINGS:` field → drop + log warning
- 2.2.6 Never create a field label not present in template

### Step 2.3 — Stage 3: Impression prompt
- 2.3.0 If Stage 2 changed-set is empty → return template IMPRESSION verbatim, no LLM call
- 2.3.1 Input = only findings marked "changed" (not raw dictation, not full report)
- 2.3.2 Output = plain text, 1–3 sentences, no field labels
- 2.3.3 Implement `src/impression.py::write_impression(changed_findings) -> str`
  - 2.3.3.1 `temperature=0` on this API call — REQUIRED from this first implementation onward,
    same reason as 2.1.4.1

### Step 2.4 — Stage 4: Validator
- 2.4.1 Check: output field-label set == template field-label set (normalized
  case-insensitive; hierarchical child labels count as fields)
- 2.4.2 Check: output field-label order == template order
- 2.4.3 Check: exactly one FINDINGS section + one IMPRESSION section
- 2.4.4 Check: no markdown fences, no extra text outside the two sections. Blank
  lines between fields, uppercased labels, numbered impression lines (`1.`) are
  ALLOWED (scorer strips list markers) — do not fail on them
- 2.4.5 Check: for every field NOT flagged as changed by Stage 2, output text is byte-identical
  to `template_content`'s text for that field (R3 enforcement — catches any Stage 2 splice bug
  before it reaches `run_val.py`)
- 2.4.6 Check: no unlabelled text exists inside FINDINGS outside recognized field spans (scan
  text between the last matched field's content and the next field label / IMPRESSION marker;
  flag if non-whitespace content remains) — catches the scorer's "unsupported unlabelled
  content" penalty case before it reaches submission
- 2.4.7 Return `(pass: bool, reason: str | None)`

### Step 2.5 — Orchestration
- 2.5.0 Fast path: if normalized dictation == `normal` (case-insensitive, strip
  punctuation/whitespace) → return `template_content` normalized to
  FINDINGS+IMPRESSION structure with zero LLM calls
- 2.5.1 `src/pipeline.py::generate_report(row) -> str`
- 2.5.2 Flow: Stage1 → Stage2 → Stage3 → assemble → Stage4 validate
- 2.5.3 On validator fail: retry Stage 1 once with failure reason appended
- 2.5.4 On second fail: log case_id + fall back to `template_content` copied unchanged as the
  report (per AGENTS.md §8, which is authoritative on conflict) — this is a valid
  FINDINGS/IMPRESSION structure, passes Stage 4, worst-case RES for that row. Do NOT hard-fail
  with no row produced: every test.csv case_id must appear exactly once in submission.csv

**CHECKPOINT 2 (must pass before Day 3):**
- [ ] pipeline runs end-to-end on 5 sample rows without crashing
- [ ] all 5 outputs pass Stage 4 validator

---

## SECTION 5 — DAY 3: MEASURE + DEBUG + IMPROVE

### Step 3.1 — Baseline run
- 3.1.1 `scripts/run_val.py`: run pipeline on full val split
- 3.1.2 Compute per-case RES + mean RES
- 3.1.3 Save (dictation, template, generated, reference, RES) to CSV

### Step 3.2 — Error analysis
- 3.2.1 Sort val results by RES descending
- 3.2.2 Manually read worst 15–20 cases
- 3.2.3 Bucket errors into:
  - (a) wrong field routing
  - (b) untouched field rewritten
  - (c) missing/garbled negation or laterality
  - (d) IMPRESSION contains unsupported info
  - (e) malformed structure
- 3.2.4 Identify highest-frequency bucket

### Step 3.3 — Targeted fixes
- 3.3.1 Routing errors dominant → add more per-modality (+label-set) field-label few-shots
  (never per-body_part alone — test has unseen body_part values)
- 3.3.2 Negation/laterality errors dominant → add explicit checklist instruction to extraction prompt
- 3.3.3 Over-rewriting still happening → audit Stage 2 code, not the prompt
- 3.3.4 (optional, time-permitting) self-consistency: run Stage 1 twice, keep better-scoring merge
- 3.3.5 Re-run `run_val.py` after each fix, confirm mean RES improved before next fix

**CHECKPOINT 3 (must pass before Day 4):**
- [ ] mean val RES improved at least once vs baseline
- [ ] two consecutive full val runs show stable (non-random) RES

---

## SECTION 6 — DAY 4: FREEZE + GENERATE + SUBMIT

### Step 4.1 — Freeze
- 4.1.1 No further prompt/logic changes after this point

### Step 4.2 — Test run
- 4.2.1 `scripts/run_submission.py`: run pipeline on all 132 test.csv rows
- 4.2.2 Run Stage 4 validator on every output
- 4.2.3 Manually review + fix any validator failures
- 4.2.4 BEFORE writing submission.csv: assert `len(output_case_ids) == len(test_case_ids) == 132`
  and no duplicate case_ids — enforces the Step 2.5.4 fallback rule actually fired for any
  failed rows rather than silently dropping them

### Step 4.3 — Submission file
- 4.3.1 Write via `pandas.to_csv` (never hand-write CSV — auto-quoting required)
- 4.3.2 Verify: 132 unique case_ids
- 4.3.3 Verify: exactly 2 columns (`case_id`, `report`)
- 4.3.4 Verify: no extra index column

### Step 4.4 — Kaggle notebook
- 4.4.0 Notebook must resolve data paths for both local (`data/*.csv`) and Kaggle
  (`/kaggle/input/*/data/*.csv`) via fallback search; install from `requirements.txt`
- 4.4.1 Port src/ + run_submission.py logic into notebooks/pipeline_notebook.ipynb
- 4.4.2 Organize by stage with markdown headers
- 4.4.3 Use `os.environ["ANTHROPIC_API_KEY"]` placeholder — never a real key
- 4.4.4 Set `temperature=0` on all LLM calls in the notebook path (submission spec requires the
  notebook to reproduce the uploaded CSV without manual case-level editing — sampling
  randomness breaks that guarantee)
- 4.4.5 Save version, share privately with `natoeaidev`

### Step 4.5 — README: design notes for hiring review
- 4.5.1 Add data disclaimer to README.md: "This benchmark uses de-identified data and must not
  be used for clinical decisions." (per competition data notice; also AGENTS.md R8)
- 4.5.2 Add "Design notes for review" section covering:
  - 4.5.2.1 Architecture choices (why Stage 2 is pure Python, why extraction is clause-level)
  - 4.5.2.2 Known failure modes / limitations found during Day 3 error analysis
  - 4.5.2.3 Reproducibility guarantees (temperature=0 throughout, no manual case-level editing)
- REASON: hiring team separately reviews clinical faithfulness, unsupported content, omissions,
  routing, reproducibility, and system design — independent of the RES leaderboard score

### Step 4.6 — Submit
- 4.6.1 Upload submission.csv to Kaggle
- 4.6.2 Paste notebook URL into submission description
- 4.6.3 Reserve buffer time for upload/formatting issues

**CHECKPOINT 4 (final, before declaring done):**
- [ ] submission.csv valid (132 rows, 2 columns, correct quoting)
- [ ] notebook shared with natoeaidev
- [ ] notebook URL in submission description
- [ ] Kaggle upload confirmed successful

---

## SECTION 7 — DEFINITION OF DONE (master checklist)

- [ ] scorer.py implemented + unit-tested (Day 1)
- [ ] 3-stage pipeline runs on all test rows without crashing (Day 2)
- [ ] validator passes 100% of generated reports (Day 4)
- [ ] mean val RES measured 2+ times with documented improvement (Day 3)
- [ ] submission.csv correct format (Day 4)
- [ ] Kaggle notebook mirrors real pipeline, no hardcoded key (Day 4)
- [ ] notebook shared + URL submitted (Day 4)
- [ ] README contains data disclaimer + "design notes for review" section (Day 4)
- [ ] submission.csv row-count/dedup assertion (132 unique case_ids) passed before upload (Day 4)
