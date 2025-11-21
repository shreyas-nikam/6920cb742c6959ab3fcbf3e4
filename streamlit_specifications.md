
# Streamlit Application Requirements Specification: Scenario-Based Model Robustness Test


## Overview

This application focuses on scenario-based robustness testing of financial ML models using structured stress transformations (T_s(x)) on input features. Instead of retraining models under every scenario, we apply transparent, parameterized shocks (e.g., macro, market, or borrower-level stresses) to the inputs and observe how the model’s predictions (f_\theta(x^{(s)})) change.

The case targets:

* **Quantitative analysts and model developers** building credit, market, or trading models.
* **Risk managers and validators** conducting SR 11-7–style robustness assessments.
* **CFA/FRM/PRM candidates** who need to connect stress testing concepts with ML implementations.

The core idea is to treat a trained model as an object under different “what-if” scenarios, where each scenario is implemented as a transformation (T_s) on the feature vector.

---

## Learning Outcomes

By completing this case, participants will be able to:

* Understand how scenario-based stress testing can be applied directly to ML models via feature-level transformations (T_s(x)).
* Translate financial scenarios (e.g., recession, credit spread widening, house price drop) into quantitative shocks on input features.
* Measure and interpret model sensitivity to structured shocks using risk-friendly metrics (e.g., change in PD, VaR, expected loss, or trading signal).
* Identify model behaviours that are overly fragile or insufficiently conservative under plausible adverse environments.
* Integrate scenario-based robustness tests into a model validation and governance workflow.

---

## Features

### Stress Transformation Library (T_s(x))

Implement a family of transformations
[
T_s: \mathbb{R}^d \to \mathbb{R}^d
]
that encode different financial scenarios. For a feature vector (x) and scenario (s):

[
x^{(s)} = T_s(x) = x \odot (1 + \delta_s),
]

where (\delta_s) is a vector of scenario shocks (e.g., +20% debt-to-income, –10% income, +200 bps credit spread), and (\odot) is elementwise multiplication.

Typical scenario types:

* **Macro downturn:** higher unemployment proxy, lower income, higher utilisation.
* **Market stress:** higher volatility, lower liquidity, wider spreads.
* **Idiosyncratic borrower shocks:** income drop, increased delinquencies, higher LTV.

### Scenario Definition Panel

Allow users to define and parameterize scenarios in terms of feature-level multipliers and shifts, for example:

```
income_scale = 0.9
utilization_scale = 1.2
house_price_scale = 0.85
```

These are then translated into the transformation (T_s(x)).

### Model Response Measurement

For a trained model (f_\theta) (e.g., logistic regression, XGBoost):

* **Baseline prediction:**
  [
  \hat{y} = f_\theta(x)
  ]

* **Stressed prediction:**
  [
  \hat{y}^{(s)} = f_\theta(x^{(s)})
  ]

Compute impact metrics such as:

[
\Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y},
\qquad
\Delta \text{EL}^{(s)} = \text{EL}^{(s)} - \text{EL},
]

where **EL** is expected loss or another aggregated portfolio measure.

### Portfolio-Level Aggregation

Roll up scenario effects from individual predictions to portfolio metrics:

* Change in mean PD or loss for a credit portfolio.
* Shift in factor exposures or VaR for an equity portfolio.
* Counts or percentages of obligors whose risk grade upgrades/downgrades under scenario (s).

### Sensitivity and Nonlinearity Analysis

Explore gradual stress paths by varying shock intensity (\alpha) in:

[
x^{(s,\alpha)} = x \odot (1 + \alpha,\delta_s), \quad \alpha \in [0,1],
]

and plotting the trajectory of model outputs (\hat{y}^{(s,\alpha)}). This reveals nonlinear thresholds and regions where the model response accelerates or becomes unstable.

## Validation-Ready Outputs

Generate tables and plots suitable for validation reports and committees:

* Scenario vs. baseline metrics
* Heatmaps by segment (e.g., rating band, sector, region)
* Distributions of (\Delta \hat{y}^{(s)}) and flags where changes exceed thresholds

---

## How It Explains the Concept

This case operationalizes model robustness and stress testing in a way that is concrete for ML:

* Instead of treating the model as a black box, each stress scenario is expressed as an explicit transformation (T_s(x)) on the input space.
* By computing (f_\theta(x^{(s)})) for a library of scenarios, participants see how predictions change when underlying economic or borrower conditions worsen.

The pseudo-code and implementation emphasize simple, interpretable transformations—such as scaling and shifting selected features—rather than complex adversarial optimization. This keeps the exercise grounded in risk management practice:

* **Risk teams** can map familiar stress narratives (“GDP down 2%, house prices –15%, spreads +200 bps”) directly to feature shocks (\delta_s).
* **Model developers** can visualise both local (per-borrower) and global (portfolio-level) sensitivity.
* **Governance forums** can compare scenario robustness across models (e.g., challenger vs. champion) using consistent metrics.

By working through this lab, participants deepen their understanding of:

* How an ML model’s mapping (f_\theta) behaves under structured, economically meaningful transformations of inputs.
* Where the model is overly sensitive, non-intuitive, or misaligned with risk appetite.

This provides a bridge between traditional scenario-based stress testing and modern ML model validation, making robustness tangible and auditable for financial institutions.
