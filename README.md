# ZeroLoop: Complaint-Text Analytics Pipeline Prototype

A prototype complaint and incident text-analytics pipeline tested on public **FDA Food Recall Enforcement Data (2012–2026)**.

---

## Important Honesty Rules
1. **The data is FDA food RECALLS, not customer complaints.** It serves as an empirical test bed for the analytical methods and architecture.
2. **Keyword labels are WEAK labels.** Accuracy measured against weak labels is circular and must never be reported as real performance.
3. **Real accuracy is measured only on a hand-labelled gold set.**

---

## Pipeline Overview

The pipeline is implemented in [`ZeroLoop_openFDA_starter.ipynb`](ZeroLoop_openFDA_starter.ipynb):

1. **Public API Ingestion (`openFDA`)**:
   - Paged year-by-year retrieval across 2012–2026 via `https://api.fda.gov/food/enforcement.json`.
   - Captures `recall_number`, `event_id`, `recall_initiation_date`, `report_date`, `recalling_firm`, `reason_for_recall`, and classification attributes.
   - Cached to `fda_food_enforcement.csv`.

2. **Data Cleaning & Date Validation**:
   - Deduplication by `recall_number` and whitespace/case normalization.
   - Initiation date validation (flagging dates later than today or after `report_date`, quarantining to `bad_dates.csv`, and falling back to `report_date` with `date_fixed=True`).

3. **Weak-Label Taxonomy**:
   - Priority-based deterministic keyword matching (*first match wins*):
     1. `allergen_undeclared`
     2. `microbial`
     3. `foreign_matter`
     4. `chemical`
     5. `labeling_packaging`
     6. `process_quality`
     7. `other`

4. **Pareto Frequency Analysis**:
   - Dual-axis Pareto chart (incident counts + cumulative percentage line).
   - Sorted descending with `other` placed last. `microbial`, `allergen_undeclared`, and `foreign_matter` together cross the 80% threshold (81.34%).

5. **Recurrence Tracking & Chronic Candidates**:
   - Firm name normalization with punctuation removal, corporate suffix stripping, leading "the " removal, and fuzzy alias resolution (`difflib`, threshold 0.93; review band 0.85–0.93 saved to `firm_alias_review.csv`).
   - True event definition based on `event_id` (fallback to normalized firm + initiation date) rather than raw product rows.
   - Detects chronic defect candidates (≥ 3 distinct events) and shortest median gaps.

6. **Gold Standard Benchmark Framework**:
   - Random sampling of 150 recalls (`random_state=42`).
   - Blind export to `gold_to_label_A.csv` and `gold_to_label_B.csv` (no weak label column to prevent anchoring bias). Reference key saved separately in `gold_key.csv`.

7. **Step 6b: Inter-Annotator Agreement**:
   - Computes Cohen's Kappa score ($\kappa$) between Annotators A and B.
   - Generates pairwise disagreement matrix and exports `disagreements.csv` for adjudication.
   - Builds `gold_labeled.csv` from consensus labels.

8. **Supervised Classifier Evaluation**:
   - Evaluates strictly against `gold_labeled.csv`.
   - Trains TF-IDF (1–2 grams, `min_df=3`, `sublinear_tf=True`) + balanced `LogisticRegression` on the non-gold training slice.
   - Compares Macro-F1 against the keyword baseline.

9. **Roadmap & Next Steps**:
   - Dense Sentence Transformers (Bi-encoders).
   - HDBSCAN unsupervised density clustering over UMAP embeddings.
   - Kaplan-Meier survival curves & Cox Proportional Hazards using `lifelines`.
   - Synthetic incident stream generator with planted anomalies.

10. **Sanity Checks**:
    - Automated assertions verifying row counts, weak-label sums, event uniqueness, date boundaries (2008 to present), and chronic event thresholds.

---

## Repository Structure

```text
├── ZeroLoop_openFDA_starter.ipynb   # Main Jupyter Notebook pipeline
├── fda_food_enforcement.csv         # Cached openFDA recall dataset (2012-2026)
├── bad_dates.csv                    # Flagged anomalous initiation date records
├── firm_alias_review.csv            # Borderline firm aliases (0.85 <= ratio < 0.93)
├── gold_to_label_A.csv              # Blind annotation sheet for Annotator A
├── gold_to_label_B.csv              # Blind annotation sheet for Annotator B
├── gold_key.csv                     # Heuristic weak-label ground key
├── build_and_run_notebook.py        # Automated notebook builder & test runner
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

---

## Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/athishio/ZeroLoop.git
cd ZeroLoop

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab / Notebook
jupyter notebook ZeroLoop_openFDA_starter.ipynb
```
