# 📱 Telco Customer Churn Analysis & Prediction

> End-to-end churn analytics on 7,043 telecom customers — combining SQL business analysis, Python ML modeling, and a 2-dashboard Tableau intelligence suite to quantify $1.67M in annual revenue at risk.

---

## 📌 Problem Statement

A telecom company is losing **1,869 customers (26.54% churn rate)** — nearly **2x the industry benchmark of 15%**. With every churned customer costing $74.44 in monthly revenue, this translates to **$1,669,570 in annualized revenue at risk**.

**Objective:** Identify the drivers of customer churn, quantify business impact in dollar terms, and build a machine learning model to predict at-risk customers before they leave — enabling proactive retention campaigns.

---

## 📊 Dataset Overview

- **Source:** [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 original features (24 after feature engineering)
- **Target:** Churn (Yes/No) — binary classification
- **Time period:** Snapshot of customer base
- **Missing values:** Cleaned in preprocessing
- **Feature categories:**
  - Demographics — gender, SeniorCitizen, Partner, Dependents
  - Account — tenure, Contract, PaperlessBilling, PaymentMethod
  - Services — PhoneService, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - Financial — MonthlyCharges, TotalCharges

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **Database** | SQL (6 analytical queries) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Preprocessing** | LabelEncoder, StandardScaler, get_dummies |
| **Modeling** | Scikit-learn, XGBoost |
| **Model Persistence** | Joblib |
| **Evaluation** | classification_report, ROC-AUC, Confusion Matrix |
| **Dashboarding** | Tableau Public (2 dashboards) |

---

## 🔄 Project Workflow

### Phase 1 — SQL Business Analysis (6 Queries)
Before Python modeling, ran 6 SQL queries to establish business baseline:

| Query | Finding |
|---|---|
| Q1: Overall Churn Rate | 26.54% (1,869 of 7,043) |
| Q2: Churn by Contract | Month-to-month 42.71% vs Two-year 2.83% |
| Q3: Revenue Impact | $139,131/month, $1,669,570/year lost |
| Q4: Churn by Tenure | 0-3 months = 56.21% churn (highest risk) |
| Q5: Churn by Internet Service | Fiber Optic 41.89% vs DSL 18.96% |
| Q6: High Risk Customers | Identified specific customer IDs at risk |

### Phase 2 — EDA in Python (6 Charts)
1. **Churn Distribution** — bar + pie showing 26.54% churn rate
2. **Churn by Contract Type** — Month-to-month vs One year vs Two year
3. **Churn by Tenure Band** — 4 tenure groups with decay curve
4. **Monthly Charges Distribution** — fraud vs retained histogram + box plot
5. **Churn by Internet Service** — Fiber Optic, DSL, No Internet
6. **Correlation Heatmap** — top features correlated with churn

### Phase 3 — Feature Engineering
- Created `TenureBand` from continuous tenure (0-12, 13-24, 25-48, 49-72 months)
- Encoded binary Yes/No columns to 1/0
- One-hot encoded multi-category variables (Contract, InternetService, PaymentMethod)
- Handled `No internet service` and `No phone service` as 0
- Final dataset: 7,043 rows × 24 features

### Phase 4 — XGBoost Model
Configuration:
```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(retained/churned),  # Handle imbalance
    random_state=42
)
```

- **Train/Test split:** 80/20 stratified (5,634 train / 1,409 test)
- **Class balancing:** scale_pos_weight handles 73:27 ratio
- **Trees built:** 200

### Phase 5 — Risk Tier Classification
Used predict_proba to bucket 1,409 test customers:

| Risk Tier | Probability Range | Customer Count |
|---|---|---|
| Low Risk | 0% – 30% | 687 |
| Medium Risk | 30% – 60% | 260 |
| High Risk | 60% – 100% | 462 |

### Phase 6 — Export & Tableau Dashboards
Exported 3 files:
- `churn_predictions.csv` (1,409 rows with risk tiers)
- `churn_encoded_dataset.csv` (full 7,043 rows encoded)
- `xgb_churn_model.pkl` (saved model)

Built 2 Tableau dashboards:
- **Dashboard 1:** Telco Churn Intelligence Dashboard (KPIs + 6 charts + 4 filters)  <img width="1395" height="882" alt="Screenshot 2026-05-22 124147" src="https://github.com/user-attachments/assets/bd7fec54-8c1b-4955-8ecd-bdc469a55f08" />

- **Dashboard 2:** Business Impact & Service Analysis (KPIs + key services + recommendations)  <img width="1394" height="884" alt="Screenshot 2026-05-22 124207" src="https://github.com/user-attachments/assets/a13f5bd4-8c6f-4142-a1c5-adcaaf78fe6c" />


---

## 🔍 Key Findings & Insights

### Finding 1 — Contract Type Is the #1 Churn Driver
| Contract | Churn Rate | Customers |
|---|---|---|
| Month-to-month | **42.71%** | 3,875 |
| One year | 11.27% | 1,473 |
| Two year | **2.83%** | 1,695 |

**Insight:** Month-to-month customers churn **15x more** than 2-year contract customers.

### Finding 2 — Early Tenure Is Highest Risk
| Tenure Band | Churn Rate |
|---|---|
| 0–3 Months | **56.21%** |
| 4–12 Months | 39.15% |
| 13–24 Months | 28.71% |
| 25–48 Months | 20.39% |
| 49+ Months | 9.51% |

**Insight:** First 3 months are critical — half of new customers churn before reaching month 4.

### Finding 3 — Fiber Optic Premium Paradox
| Service | Churn Rate | Avg Monthly Charges |
|---|---|---|
| Fiber Optic | **41.89%** | $91.50 |
| DSL | 18.96% | $58.10 |
| No Internet | 7.40% | $21.08 |

**Insight:** Fiber Optic is premium-priced but has highest churn — clear service quality issue.

### Finding 4 — Revenue at Risk
- Monthly revenue lost: **$139,131**
- Annual revenue lost: **$1,669,570**
- Avg revenue per churned customer: **$74.44/month**
- Total historical value lost: **$2,862,927**
- Avg tenure before leaving: **18 months**

### Finding 5 — Monthly Charges Correlate with Churn
- Churned customers pay **$74.44/month** average
- Retained customers pay **$61.27/month** average
- **$13.17 difference** — higher charges drive churn

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| **Class imbalance (26.54% churn vs 73.46% retained)** | Used `scale_pos_weight` in XGBoost to weight classes inversely |
| **TotalCharges had blank strings instead of nulls** | Converted to numeric with `errors='coerce'` then filled |
| **Mixed encoding (binary + multi-category)** | Hybrid approach: map() for binary, get_dummies() for multi-category |
| **SeniorCitizen object dtype caused XGBoost error** | Converted all object dtypes to numeric before training |
| **Dashboard initially too cluttered in one page** | Split into 2 focused dashboards by audience (analyst vs executive) |
| **Tableau couldn't read encoded one-hot columns as categories** | Created calculated fields (Contract Type, Internet Service Type) to decode |

---

## 💼 Business Recommendations

1. **Deploy first-90-days onboarding program** — Half of 0–3 month customers churn; intensive engagement here saves the most
2. **Incentivize contract upgrades** — Offer 10–15% discount for month-to-month → annual conversion (churn drops 15x)
3. **Investigate Fiber Optic service quality** — 41.89% churn at premium pricing signals reliability or support gaps
4. **Promote auto-pay enrollment** — Electronic check users churn 3x more than credit card users
5. **Bundle protection services** — Customers without Tech Support and Online Security churn at 31%+
6. **Use risk tier model for retention prioritization** — Focus retention budget on 462 high-risk customers first

---

## 🚀 Future Scope

- Build customer lifetime value (CLV) model alongside churn — high-value churn = bigger loss
- Add survival analysis to predict **when** customers will churn, not just **if**
- Implement SHAP values for individual customer churn explanations
- Build a recommendation engine for retention offers (which discount works best per segment)
- Real-time churn risk scoring API for customer service team
- Quarterly churn dashboard refresh with new data
- A/B test retention campaigns against high-risk tier predictions

---

## 📂 Project Structure

```
Customer Churn Analysis/
├── churn_analysis.ipynb              # Full Python notebook
├── telco_churn_cleaned.csv           # Source dataset (7,043 rows)
├── churn_encoded_dataset.csv         # Encoded for modeling
├── churn_predictions.csv             # 1,409 test predictions with risk tiers
├── xgb_churn_model.pkl               # Saved XGBoost model
├── SQL_Queries/
│   ├── q1_churn_rate.csv
│   ├── q2_churn_by_contract.csv
│   ├── q3_revenue_impact.csv
│   ├── q4_churn_by_tenure.csv
│   ├── q5_churn_by_service.csv
│   └── q6_high_risk_customers.csv
├── charts/                           # 6 EDA chart PNGs + model evaluation
└── chun.twb                          # Tableau workbook (2 dashboards)
```

---

## 📬 About / Contact

**Hari Ram** — Data Analyst | Hyderabad
B.Tech CSE, CVR College of Engineering (2024) | AI Variant Intern

- 📧 **Email:** ramh2125@gmail.com
- 📍 **Location:** Hyderabad, India
- 💼 **Open to:** Data Analyst / Junior Data Scientist roles

