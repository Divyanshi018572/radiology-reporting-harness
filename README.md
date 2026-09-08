# Radiology Reporting Harness

This benchmark uses de-identified data and must not be used for clinical decisions.

## Data notes

- `train.csv`: 636 rows x 9 cols (inputs + `report`). `test.csv`: 132 rows x 8 cols.
- Modalities train: XRAY 380 / MRI 158 / CT 71 / USG 27.
- 183 unique template field labels (hierarchical, e.g. `Menisci:` -> `Medial meniscus:`).
  Labels vary in case (`Osseous Structures:` vs `OSSEOUS STRUCTURES:`) — match
  case-insensitively, emit template's original text.
- Dictation bimodal: 1-word `normal` (copy template) vs long detailed (up to ~418 words).
- `OTHER FINDINGS:` in 56/636 train, 28/132 test — routing there matters.
- Test `body_part` has unseen values (Arm, Heel, Thumb, Bladder, Paranasal sinus,
  Orbit, Sternum, Knee patella, Ankle calcaneous) — route by modality + label set,
  not by `body_part`.
- Reference formatting is free (blank lines, uppercased labels, numbered impression
  `1. 2.`) — scorer strips list markers, validator must allow them.

## Design notes for review

- Architecture choices: Stage 2 merge is pure Python (never LLM) so untouched
  fields/clauses survive verbatim; extraction is clause-level
  (`matched_template_span` + `new_clause_text`) for compound-sentence preservation.
- Known failure modes: hierarchical label mismatch, long-dictation routing,
  negation/laterality drops, impression adding unsupported claims.
- Reproducibility: `temperature=0` on all LLM calls from first implementation,
  no manual case-level editing, cached LLM responses, seeded train/val split.
