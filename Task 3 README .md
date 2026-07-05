# TASK 3: CREDIT RISK ANALYSIS - PROBABILITY OF DEFAULT PREDICTION
## Complete Documentation

**Status:** ✅ COMPLETE  
**Date:** June 22, 2026  
**For:** JPMorgan Chase Retail Banking Risk Team

---

## Executive Summary

### Problem Statement
The retail banking arm has experienced higher-than-expected default rates on personal loans. Need to build a predictive model that estimates the probability of default (PD) for borrowers based on their characteristics.

### Solution
Built and compared 4 machine learning models to predict probability of default:
1. **Logistic Regression** (ROC AUC: 1.00) ← RECOMMENDED
2. Gradient Boosting (ROC AUC: 0.9999)
3. Random Forest (ROC AUC: 0.9997)
4. Decision Tree (ROC AUC: 0.9997)

### Expected Loss Framework
```
Expected Loss (EL) = PD × LGD
Where:
  PD = Probability of Default (model prediction)
  LGD = Loss Given Default = 1 - Recovery Rate
  Recovery Rate = 10% → LGD = 90%

Example:
  High-risk borrower: PD = 99% → EL = 89%
  On $10,000 loan → Expected loss provision = $8,900
```

---

## Data Overview

### Dataset: Task_3_and_4_Loan_Data.csv
- **Records:** 10,000 borrowers
- **Features:** 6 predictive variables
- **Target:** Binary (default = 1, non-default = 0)
- **Default Rate:** 18.51% (realistic for subprime lending)

### Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `credit_lines_outstanding` | Integer | Number of active credit lines | 0-5 |
| `loan_amt_outstanding` | Continuous | Current loan balance | $1,959-$5,579 |
| `total_debt_outstanding` | Continuous | Total debt (all creditors) | $1,756-$25,000 |
| `income` | Continuous | Annual income | $23,448-$98,689 |
| `years_employed` | Integer | Years at current job | 1-7 |
| `fico_score` | Integer | Credit score | 408-850 |

### Target Variable
- **default** = 1: Borrower defaulted on loan
- **default** = 0: Borrower did not default

**Output**

<img width="1056" height="577" alt="image" src="https://github.com/user-attachments/assets/003350df-56f3-4041-ad27-645862c4a226" />

---

## Feature Analysis

### Key Insights

**1. Credit Lines Outstanding** (STRONGEST PREDICTOR)
- Defaulters average: 2.5 credit lines
- Non-defaulters average: 1.2 credit lines
- Interpretation: More credit lines = higher default risk
- Impact: Single most important feature in all models

**2. FICO Score** (PROTECTIVE FACTOR)
- Defaulters average: 597
- Non-defaulters average: 655
- Interpretation: Higher FICO = lower default risk
- Impact: Standard credit metric, strong predictor

**3. Income** (PROTECTIVE FACTOR)
- Defaulters average: $42,000
- Non-defaulters average: $59,000
- Interpretation: Higher income = ability to repay
- Impact: 40% income difference between groups

**4. Years Employed** (PROTECTIVE FACTOR)
- Defaulters average: 2.8 years
- Non-defaulters average: 3.9 years
- Interpretation: Job stability reduces default risk
- Impact: Signals income stability

**5. Total Debt Outstanding**
- Defaulters average: $12,500
- Non-defaulters average: $6,500
- Interpretation: High debt burden increases risk
- Impact: Debt-to-income ratio is important

**6. Loan Amount Outstanding**
- Less predictive than other features
- Varies widely in both groups
- Impact: Weak predictor individually

---

## Models Compared

### 1. LOGISTIC REGRESSION
**Type:** Linear classification  
**Interpretability:** High (coefficients show feature impact)  
**ROC AUC:** 1.0000 ← BEST

**Advantages:**
- Simple, interpretable model
- Outputs probabilities naturally (0-1 range)
- Fast training and prediction
- Perfect performance on this dataset
- Easy to explain to stakeholders

**How It Works:**
```
log(odds of default) = β₀ + β₁×credit_lines + β₂×income + ...

Where coefficients show:
  - β > 0: Increases default probability
  - β < 0: Decreases default probability
```

**Feature Coefficients:**
- credit_lines_outstanding: +8.75 (strong risk factor)
- total_debt_outstanding: +3.84 (risk factor)
- years_employed: -2.90 (protective)
- income: -2.35 (protective)
- fico_score: -1.14 (protective)

**Recommendation:** Use this model for interpretability and performance

---

### 2. GRADIENT BOOSTING
**Type:** Ensemble (sequential decision trees)  
**Interpretability:** Medium (feature importance available)  
**ROC AUC:** 0.9999

**Advantages:**
- Highest predictive accuracy
- Handles non-linear relationships
- Robust to outliers
- Strong on complex patterns

**Disadvantages:**
- Slower training time
- Less interpretable
- Risk of overfitting

---

### 3. RANDOM FOREST
**Type:** Ensemble (parallel decision trees)  
**Interpretability:** Medium  
**ROC AUC:** 0.9997

**Advantages:**
- Fast parallel training
- Handles non-linearity
- Feature importance rankings

**Disadvantages:**
- Less interpretable than logistic regression
- Slightly lower ROC AUC

---

### 4. DECISION TREE
**Type:** Single decision tree  
**Interpretability:** High (visual tree structure)  
**ROC AUC:** 0.9997

**Advantages:**
- Very interpretable (can visualize decision path)
- Fast prediction
- No feature scaling needed

**Disadvantages:**
- Prone to overfitting if not constrained
- Less stable than ensembles


** Model Comparision Output**

<img width="941" height="557" alt="image" src="https://github.com/user-attachments/assets/6e7ad8ee-8c6d-4018-811c-4e9156f13886" />

---

## Model Performance

### Test Set Metrics

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 99.90% | 100.00% | 99.46% | 99.73% | **1.0000** |
| Gradient Boosting | 99.75% | 99.46% | 99.19% | 99.32% | 0.9999 |
| Random Forest | 99.35% | 98.11% | 98.38% | 98.25% | 0.9997 |
| Decision Tree | 99.55% | 99.18% | 98.38% | 98.78% | 0.9997 |

### What These Metrics Mean

**Accuracy:** Percentage of correct predictions
- All models >99% → Excellent overall performance

**Precision:** Of predicted defaults, how many actually defaulted?
- 100% (Logistic) → No false alarms

**Recall:** Of actual defaults, how many did we catch?
- 99.46% (Logistic) → Catches almost all defaulters

**ROC AUC:** Area under the ROC curve (0-1 scale, 1 = perfect)
- All models ≥0.9997 → Excellent discrimination

---

## Expected Loss Calculation

### Formula
```
Expected Loss = PD × LGD

Where:
  PD = Probability of Default (0-1)
  LGD = Loss Given Default (1 - Recovery Rate)
  Recovery Rate = 10% (assumption)
  LGD = 90%

Dollar Loss = EL × Loan Amount
```

### Examples

**Low-Risk Borrower**
```
Income: $80,000 | FICO: 750 | Credit Lines: 1 | Debt: $5,000

Logistic Regression:
  PD: 0.00%
  EL: 0.00%
  On $3,000 loan: $0 expected loss
```

**Medium-Risk Borrower**
```
Income: $45,000 | FICO: 650 | Credit Lines: 3 | Debt: $15,000

Logistic Regression:
  PD: 98.08%
  EL: 88.27%
  On $6,000 loan: $5,296 expected loss
```

**High-Risk Borrower**
```
Income: $30,000 | FICO: 550 | Credit Lines: 5 | Debt: $25,000

Logistic Regression:
  PD: 100.00%
  EL: 90.00%
  On $8,000 loan: $7,200 expected loss
```

---

## Implementation Guide

### Using the Credit Risk Model

#### Jupyter Notebook
```bash
jupyter notebook Task_3_Credit_Risk_Analysis.ipynb
# Interactive model building and testing
```


## Validation & Testing

### Test Set Performance
- 2,000 holdout samples (20% of data)
- Stratified sampling maintains 18.5% default rate
- Models achieve 99.9% accuracy

### Cross-Validation
- 5-fold cross-validation confirms stability
- Consistent performance across folds
- No evidence of overfitting

### Backtesting
Recommend backtesting on recent loan cohorts to verify:
- Predicted PD vs actual default rates
- Model calibration (are PD estimates accurate?)
- Any drift in population characteristics

---

## Integration Points

### 1. Loan Approval Workflow
```
Loan Application
    ↓
Feature Extraction
    ↓
Risk Model (PD Prediction)
    ↓
Expected Loss Calculation
    ↓
Risk Decision (Approve/Decline)
```

### 2. Pricing & Rate Setting
```
PD Estimate
    ↓
Risk Premium Calculation
  (Interest Rate Spread)
    ↓
Loan Pricing
```

### 3. Portfolio Risk Management
```
Aggregate PD across portfolio
    ↓
Total Expected Losses
    ↓
Regulatory Capital Requirements
    ↓
Provision Setting
```

### 4. Collection & Workout Strategy
```
High PD Borrowers
    ↓
Enhanced Monitoring
    ↓
Proactive Collection Outreach
    ↓
Early Default Prevention
```

---

## Key Findings & Recommendations

###  Model Selection
**Recommendation: Use Logistic Regression**

**Rationale:**
1. Perfect ROC AUC (1.0000)
2. Highest interpretability
3. Most explainable to stakeholders
4. Fastest prediction time
5. Standard in risk management

**Fallback:** Use Gradient Boosting if additional 0.0001 AUC is critical

###  Risk Factors (by importance)
1. **Credit lines outstanding** - Strongest predictor
2. **Total debt outstanding** - Debt burden
3. **Years employed** - Job stability
4. **Income** - Repayment capacity
5. **FICO score** - Historical credit behavior

###  Business Actions
1. **Underwriting:** Weigh credit lines heavily in decisions
2. **Pricing:** Add risk premium for high PD borrowers
3. **Monitoring:** Watch borrowers with 3+ credit lines
4. **Collections:** Proactive outreach for PD > 50%
5. **Capital:** Set provisions based on portfolio PD

---

## Limitations & Assumptions

### Model Limitations
- ⚠️ Trained on specific population (may not generalize)
- ⚠️ No macroeconomic variables (recession effects)
- ⚠️ No borrower behavior data (payment history)
- ⚠️ 10% recovery rate is assumption (varies by collateral)

### Data Limitations
- ⚠️ 10,000 samples (small for deep learning)
- ⚠️ 6 features (limited information)
- ⚠️ Single time period (no seasonal effects)
- ⚠️ No borrower demographics (age, industry)

### Recommendations for Improvement
1. Add macroeconomic indicators (unemployment, rates)
2. Include payment behavior (late payments, utilization)
3. Add borrower demographics (age, industry, location)
4. Expand to 100K+ samples for deep learning
5. Model recovery rates (not constant 10%)
6. Regular model retraining (quarterly)

---

## Production Deployment Checklist

- [✓] Model built and validated
- [✓] Test set performance documented

---

## FAQ

**Q: What's a "good" probability of default?**
A: Depends on pricing:
- PD < 5%: Very low risk (prime lending)
- PD 5-20%: Low-medium risk (auto loans)
- PD 20-50%: Medium risk (personal loans)
- PD > 50%: High risk (subprime/workout)

**Q: Why use 10% recovery rate?**
A: Conservative assumption. Actual varies:
- Secured loans: 70-90% recovery
- Personal loans: 5-30% recovery
- Unsecured: <5% recovery

**Q: How often should I retrain?**
A: Quarterly recommended:
- Monitor ROC AUC monthly
- Retrain if AUC drops >2%
- Annual retraining at minimum

**Q: What if PD > 100%?**
A: Capped at 1.0 (100%). Model outputs clipped.

---

**Problem Solved On:** June 22, 2026  
**Status:** Production Ready  
**Contact:** JPMorgan Chase Quantitative Research
