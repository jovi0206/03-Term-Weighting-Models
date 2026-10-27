# NCKU Biomedical Information Retrieval — Project 3

**Topic:** Term Weighting / TF-IDF / Vector Space / Retrieval Evaluation  
**Course due date:** 2026-10-27  
**Edition:** Independent modular Project 3 SRC (M01–M09 + E2/E3/E4/E7/E8), now six-page UI with Batch 2 lexical comparison and Batch 3 read-only Word2Vec, checked in Batch 4 (2026-10-21).  
**Source boundary:** The user's uploaded Project 2 Extension ZIP and the 2026-10-14 initial Project 3 system were read-only inputs; neither is modified.

## Run on Windows (PowerShell; from `03-term weighting models`)

**Step 1 — Create the environment (once):**

```powershell
py -3.12 -m venv "D:\venv\project3_term_weighting"
& "D:\venv\project3_term_weighting\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
```

**Step 2 — Verify the core first:**

```powershell
python src\project3_core_selftest.py
python -m pytest -q tests\test_project3.py
```

`pytest` is included in requirements. After applying the Batch 4 regression patch, `python -m pytest -q tests` should collect 33 tests in the reference suite (including six professor-follow-up tests). The self-test itself does not depend on pytest.

**Step 3 — Generate official Condition C reproducible results:**

```powershell
python src\project3_prepare_all.py --condition C
```

**Step 4 — Open the six-page app:**

```powershell
python -m streamlit run src\m09_app.py
```

Keep all venvs **outside OneDrive** at `D:\venv\<environment-name>`. Home and office each create/install a separate local environment even though their path is identical. If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for this terminal, or run `D:\venv\project3_term_weighting\Scripts\python.exe` directly. Never copy a previous Project `.git` folder or virtual environment here.

## What is here?

```text
03-term weighting models/
├── src/
│   ├── m01_corpus_reader.py               load frozen GLP-1 abstracts
│   ├── m02_text_processor.py              A/B/C/D token policies
│   ├── m03_index_builder.py               sparse document-term matrices, TF/CF/DF/IDF
│   ├── m04_query_processor.py             query vector in same vocabulary
│   ├── m05_similarity.py                  cosine / dot / L2 dot
│   ├── m06_retrieval_engine.py            deterministic Top 10 ranking
│   ├── m07_result_processor.py            explanations and term statistics
│   ├── m08_controller.py                  stable lightweight facade
│   ├── m09_app.py                         Streamlit UI, six horizontal pages + 2-D plot
│   ├── e2_01_porter.py                    unchanged Project 2
│   ├── e2_02_preprocessing_experiment.py  unchanged Project 2
│   ├── e4_02_term_statistics.py           unchanged Project 2
│   ├── e3_01_word2vec_viewer.py          optional pretrained Project 2 model viewer (read-only)
│   ├── e4_03_term_weighting.py            5 weighting policies; single registry
│   ├── e4_04_two_term_comparison.py     TF/DF/IDF two-term source
│   ├── e4_05_weighting_comparison.py     query/document five-weight evidence
│   ├── e7_01_retrieval_evaluation.py      relevance judgments + IR metrics
│   ├── e7_02_weighting_experiments.py     ranking and normalization comparisons
│   ├── e8_01_vector_projection.py         TruncatedSVD 2-D doc/query visualization
│   ├── project3_config.py                paths, preprocessing, score settings
│   ├── project3_contracts.py             IndexData data interface
│   ├── project3_artifacts.py             reusable CSV/JSON writers
│   ├── project3_prepare_all.py           reproducible official artifacts
│   └── project3_core_selftest.py          no-pytest core checks
├── data/query_relevance_rules.json       10 rule-based *proxy* qrels definitions
├── prepared_data/glp1_1000_abstracts.jsonl.gz  unchanged Project 2 corpus
├── results/                               generated ranking/metric CSV/JSON
├── tests/                                33 reference regression cases over five test modules
├── docs/                                  lineage, learning, teacher RQs and test report
├── assignment/                            optional local PDF only; NOT included in this ZIP or Git
├── artifacts/embedding_experiments/glp1/  trusted locally copied Word2Vec model files
├── requirements.txt
└── .gitignore
```

**Not required here:** Project 2 original raw `Project3/data/glp1_*` XML folders are kept outside this repository by the user. They are valuable provenance, but the runtime uses the frozen prepared corpus. There is no new PubMed download. `gensim` loads the original, locally copied Project 2 Extension Word2Vec models for a **separate teaching panel**; they do not modify TF-IDF search.

## Six Streamlit pages (horizontal selector under the green header)

1. **01 Retrieval:** enter a Query, select one of 5 weights and Cosine/Raw Dot/L2 Dot, show actual Top10 scored abstracts with term-level evidence.
2. **01-1 Weighting Comparison:** A1/A2/A3 exact per-term weighting and query/document comparisons; separate right-hand read-only pretrained CBOW/Skip-gram Top-K and word-word Cosine.
3. **02 Term Weighting:** TF/CF/DF/IDF analysis, per-document weights and L2 norms.
4. **03 Controlled Experiments:** compare same Query under two weighting methods; raw dot vs cosine vs explicit L2 dot.
5. **04 Evaluation:** 10 fixed topical queries, proxy qrels, P@10 / Recall@10 / MRR@10 / nDCG@10, CSV export.
6. **05 Vector Projection:** TruncatedSVD converts L2-normalized high-dimensional document vectors to 2-D; plot 1,000 docs, projected Query, original Cosine Top 10 overlays, hover PMID/title/original score, and CSV export. The chart is **NOT used to rank**. See `docs/UI_2D_EXTENSION.md`.

The **Preprocessing A–D** control is now in a main-page expander; there is no navigation sidebar. The current tab is retained in Streamlit session state when interacting with widgets. `st.segmented_control` is used on supported versions, otherwise a horizontal `st.radio` is used.

Formal experiment configuration is **1000 documents, Condition C, 5 methods, 10 queries, Cosine, k=10**. The IDF formula is unsmoothed `log10(N/DF)`, matching the unchanged Project 2 E42 implementation. Query and document use the same weighting policy and vocabulary.

The `results/` root is reserved for formal C. Running with `--condition A`, `B`, or `D` writes to `results/conditions/A` etc., so exploratory runs **do not overwrite** formal C output.

## Expansion without rewriting the core

- **New term weighting:** add a focused E4 implementation and register the callable once in `e4_03_term_weighting.py` under `WEIGHTING_FUNCTIONS`. M03/M04/M09 and the evaluation automatically use its `WEIGHTINGS` names. Verify with self-test before reporting new results.
- **New retrieval experiment:** add an `e7_0N_*.py` module that receives the IndexData-compatible facade and uses `index.search()`; do not put new math into M09.
- **A new project is NOT required to reuse this M01–M09 structure.** Assess the next homework requirements first, selectively reuse files, and record the source/edits/dependencies. Do not bulk-import Project 2 or Project 3 modules without need.

## Provenance, limitations, GitHub

- `docs/SRC_MODULE_LINEAGE.md`: exact source-to-target map and module dependencies.
- `docs/BUILD_PROVENANCE.md`: original corpus/legacy module checksum, excluded original project components.
- `docs/LEARNING_HANDOFF.md`: flow and teacher RQ1–RQ5 for Learning Line.
- `docs/TEST_RESULTS.md`: real pass/limitations.
- `.gitignore` excludes `.venv/`, Python cache and `assignment/`. It **includes** `src/`, `prepared_data/`, `data/`, `results/`, `docs/` to make offline reproduction possible.

**Relevance-warning:** These 10 per-query labels are thematic regex **proxies** because the provided corpus has no external relevance ground truth. They were not manually judged and may favor lexical matching. Document this limit in class.

**Verification boundary:** Python core, 2-D numeric projection, reports, tests and regression scores have been checked in the build environment. Streamlit UI **has not been interactively launched** here (Streamlit unavailable). Confirm the actual top navigation, plot hover and CSV on the user's Windows PC before claiming end-to-end UI verification.

## Evaluation chart / Learning Line update (2026-10-14)

`04 Evaluation` now uses **Plotly grouped bars**: four distinct IR metrics on the X axis, five weighting methods side-by-side for each metric, fixed 0–1 Y axis. It no longer stacks unrelated scores. `src/project3_charts.py` is presentation-only, not a retrieval algorithm; the E7 evaluation inputs, CSV, Qrels and metrics are unchanged.

The current `requirements.txt` includes `plotly`, `scikit-learn`, `pytest`, and `gensim` for the optional pretrained embedding viewer. After updating existing files, run `python -m pip install -r requirements.txt`; no venv rebuilding or corpus downloading required. The latest complete **System → Learning Line** handoff is `docs/LEARNING_LINE_PASTE.md`.

## Batch 4 (2026-10-21) — current validation and learning handoff

- Read `docs/BATCH4_SYSTEM_REGRESSION_20261021.md` for **33/33 local reference tests**, core self-test, remaining Windows six-page manual checklist and why the previous Windows suite had 27 tests.
- Read `docs/BATCH4_RELEVANCE_JUDGMENT_20261021.md` before presenting **04 Evaluation**. The automated proxy labels can confuse clinical GLP-1 with *C. elegans* GLP-1/Notch (`PMID 26833730`); they are not judged ground truth.
- Read `docs/BATCH4_LEARNING_LINE_HANDOFF_20261021.md` for the six-page flow, RQ1–RQ5, Week 2 follow-ups and Word2Vec teaching boundaries.
- The Batch 4 patch updates only `README.md`, adds/aligns `tests/test_teacher_followup.py`, and adds these three docs. **No core SRC or official results are edited.**
- Some earlier dated docs in `docs/` intentionally describe historical intermediate versions (five pages / no Word2Vec); the newest Batch 4 handoff and this README describe the current version.
