# Pharmacy Claims & Prior Auth Dashboard (Portfolio Project)

## Goal
Build an end-to-end analytics project that tracks **claim outcomes** and **prior authorization (PA) flow** to surface:
- rejection drivers
- turnaround time bottlenecks
- cost/patient impact proxies

## Why it matters
Pharmacies lose time and revenue due to avoidable rejects (e.g., coverage, refill-too-soon, PA required). This dashboard shows where the friction lives and what to fix.

## Data
**Phase 1 (Synthetic):**
- Generate realistic claim + PA data (to avoid PHI)
- Columns: date, drug_class, payer, rejection_code, outcome, days_to_resolve, copay_bucket, store_workload_proxy

**Phase 2 (Optional Real-ish):**
- Use public datasets as enrichment (drug class lists, etc.)

## Key Questions
1. What are the top rejection codes and how do they change over time?
2. Which payers/drug classes drive the highest reject rates?
3. What’s the median time-to-resolution for PA-required claims?
4. What’s the workload impact (claims touched per successful fill)?

## Metrics
- Rejection rate (%)
- Approval rate (%)
- Median days to resolve
- Volume by payer/drug class
- “Touches” proxy per claim (synthetic)

## Deliverables
- Jupyter notebook: EDA + cleaning
- Dataset generator script (synthetic data)
- Dashboard (Tableau / Power BI / Streamlit)
- Final report (PDF): insights + recommendations

## EDA Script
Use the standalone script to generate three plots and a short insights summary.

```bash
python scripts/eda.py --data-path data/synthetic_claims.csv --output-dir outputs/eda
```

**Outputs**
- `outputs/eda/rejection_rate_by_payer.png`
- `outputs/eda/top_rejection_codes.png`
- `outputs/eda/days_to_resolve_by_outcome.png`
- `outputs/eda/insights.md`

## Baseline Model (v1)
Train a baseline classifier to predict whether a claim will be rejected.

```bash
python scripts/baseline_model.py --data-path data/synthetic_claims.csv --output-dir outputs/model_v1
```

**Outputs**
- `outputs/model_v1/baseline_model_metrics.json`
- `outputs/model_v1/baseline_model_metrics.md`
- `outputs/model_v1/baseline_model_roc_curve.png`

## Tech Stack
Python (pandas), SQL (optional), visualization (Tableau/Power BI/Streamlit)

## Next Steps (This Week)
- [ ] Create synthetic dataset generator
- [ ] Run EDA notebook
- [ ] Define dashboard wireframe (4 charts + 2 KPI tiles)

## Project Status
Started: [today’s date]
