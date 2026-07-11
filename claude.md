<!-- Purpose: Conventions that Claude Code must follow when generating or modifying code in this project. -->

# Project Conventions

This document defines the rules Claude Code must follow when working in this project. Read it before making any change.

## 1. Records and Changelog

Amended 2026-07-11. Process records stay local; `CHANGELOG.md` is the published update history.

### 1.1 Records (local, not version-controlled)

The project root contains a `records` folder for logging every plan and update. It is listed in
`.gitignore` and does not ship with the working tree. Records written before 2026-07-11 remain in
git history and can be read with `git log -- records/`.

1. For each plan or update, create a new `.md` file in the `records` folder.
2. Filename format: `time_action_content`.
   - `time` uses the `YYYY-MM-DD-NN` format, where `NN` is a two-digit sequence number starting from `01` that indicates which change of that day it is (there may be several on the same day).
   - `action` is one of: `plan`, `add`, `debug`, `refactor`, `proofread`, `test`, `audit`.
   - `content` briefly states what is being done. Create the file with the plan when the work begins, and fill in the implementation and results once it is finished.
3. The body must cover three parts: Goal, Result, and Follow-up.

### 1.2 Changelog (published)

`CHANGELOG.md` is the project's outward-facing update history. **Every time a record is completed,
add one line to `CHANGELOG.md` in the same commit.** A record without its changelog line is
unfinished work.

1. Sections are dated (`## YYYY-MM-DD`), newest first. The `##` heading is also the anchor other
   documents link to (`CHANGELOG.md#2026-07-11`).
2. One line per record, prefixed by its ID and action: ``- `2026-07-11-02` test — …``.
3. The line states what was done and what the conclusion was, in one sentence, with the numbers if
   there are any (e.g. "clean-fid 11.226, within the pre-set gate of ≤20"). It does not narrate process.
4. When a record's content is cited from `README.md` or `docs/`, cite the changelog anchor, not the
   record path — the record path does not exist in the published tree.

### 1.3 Outward-facing documents (version-controlled)

Pre-registrations, protocol amendments, and final verdicts are not process records. They are
outward-facing evidence and must live under `docs/` and be committed **before** the run they govern.
The CIFAR-100 pre-registration (`docs/prereg_cifar100.md`) is the reference example. See §5.

## 2. File Header Comments

Every file in the project must begin with a short comment describing the file's purpose.

## 3. Language and Writing Style

1. `README.md`, the `.md` files in the `records` folder, code comments, analysis and conclusions, and commit messages must be written primarily in Traditional Chinese.
2. Use plain, direct prose. Avoid unnecessary adjectives or modifiers.
3. Do not use symbols as status markers, such as check marks, crosses, or warning icons.
4. Keep naming consistent across the whole project for variables, functions, files, and folders. Use one name per concept and avoid mixing different terms for the same thing.

## 4. Minimal Change Principle

1. **MVP first, build up incrementally.** Make the minimal working version fully correct before adding anything on top. It is better to support one fewer sampling configuration than to leave the whole project half-finished.
2. **STOP after each stage.** Present that stage's charts and data to the author, and proceed to the next stage only after the author confirms. Do not run through all stages in one pass.
3. **For features outside the spec, ask first; do not add them on your own.** Any extension not listed in this document (a new sampler, a new metric, a new dataset scale, or a new visualization) must be confirmed with the author before implementation.
4. **Code must be readable and commented to explain the modeling and measurement rationale.** Assume the reader is a first-year graduate student new to diffusion models: for each key step, explain what the step does and why.

## 5. Freeze and Metadata Conventions

Added 2026-07-09 (E5, execution directive `R-2026-07-08-02`/`R-2026-07-09-04`); §5.1 amended 2026-07-11 to follow the records/changelog split in §1. Convention-level only; does not touch the pre-registration protocol.

1. **Freeze definition.** A "frozen" specification is not frozen by a record header alone. It is frozen when all four hold: (a) the rule is written in prose in a **version-controlled** document — a pre-registration or amendment under `docs/` (§1.3), not a local record, since a freeze that only exists on the author's disk cannot be verified as predating the run; (b) it is expressed in committed code where a computation is involved; (c) it passes a dry-run on the already-unblinded data before the real run; and (d) the run's output carries a hash or byte-level reconcile against the frozen target. Persistence-pass (P) assets are the basis for (d). The commit carrying (a) must precede any sampling the specification governs.
2. **Driver metadata completeness.** Every measurement driver must record, in its output metadata, all parameters needed to reproduce the numbers: `start_timestamp`, full `argv`, every analysis parameter (e.g. `nearest_k` and the effective `k = min(k, n-1)`, `tau_fraction`, `batch`), and the environment versions (torch / cuda / cudnn). Rationale: the P0 source-tracing incident arose because `nearest_k` was not stored; a scalar that cannot be traced to its parameters cannot be reconciled.
