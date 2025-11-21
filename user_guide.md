id: 6920cb742c6959ab3fcbf3e4_user_guide
summary: Scenario-Based Model Robustness Test - Clone User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Understanding ML Model Robustness through Scenario Testing

## 1. Introduction to Scenario-Based Robustness Testing
Duration: 0:05:00

Welcome to QuLab, your interactive guide to understanding scenario-based robustness testing for financial Machine Learning (ML) models. In today's dynamic financial landscape, simply building predictive models is not enough; we must also understand how they behave under various adverse, yet plausible, conditions. This codelab will guide you through a powerful methodology that applies structured stress transformations, denoted as $T_s(x)$, directly to the input features of a trained ML model. The core idea is to evaluate how a model's predictions, $f_\theta(x^{(s)})$, change when exposed to transparent and parameterized shocks, without the need for retraining the model for every new scenario.

This approach is indispensable for:
*   <b>Quantitative analysts and model developers</b> in credit, market, or trading model development, who need to ensure their models are robust.
*   <b>Risk managers and validators</b>, performing crucial robustness assessments aligned with regulatory standards like SR 11-7, which mandates thorough model validation including stress testing.
*   <b>CFA/FRM/PRM candidates</b>, providing a practical bridge between theoretical stress testing concepts and their real-world ML implementations.

The fundamental principle we explore is treating a pre-trained model as an object that is subjected to various "what-if" scenarios. Each scenario is formally defined as a transformation $T_s$ applied to the model's feature vector. This helps us understand how a model behaves under structured, economically meaningful transformations of inputs.

<aside class="positive">
<b>Learning Outcomes:</b> By completing this codelab, you will be able to:

*   Understand how scenario-based stress testing can be applied directly to ML models via feature-level transformations ($T_s(x)$).
*   Translate financial scenarios (e.g., recession, credit spread widening, house price drop) into quantitative shocks on input features.
*   Measure and interpret model sensitivity to structured shocks using risk-friendly metrics (e.g., change in PD, VaR, expected loss, or trading signal).
*   Identify model behaviours that are overly fragile or insufficiently conservative under plausible adverse environments.
*   Integrate scenario-based robustness tests into a model validation and governance workflow.
</aside>

## 2. Exploring the Stress Transformation Library
Duration: 0:07:00

In this step, we will explore the foundational concept of the Stress Transformation Library, which is a key component of our scenario-based testing framework.

<aside class="positive">
<b>Navigation:</b> In the Streamlit application's sidebar, select "<b>Stress Transformation Library</b>".
</aside>

This page introduces the concept of how financial scenarios are mathematically represented as transformations applied to your model's input features. For any given feature vector $x$ (representing, for example, a customer's financial profile), and a specific scenario $s$, we calculate a *transformed* (or *stressed*) feature vector $x^{(s)}$ using the following general form:

$$ x^{(s)} = T_s(x) = x \odot (1 + \delta_s) $$

Let's break down what each component means:
*   $x$: This is your original, baseline feature vector. It contains all the input values your ML model would normally use to make a prediction.
*   $\delta_s$: This is a vector of *scenario shocks*. Think of it as a list of changes or impacts that the scenario $s$ has on each individual feature. For instance, if a recession scenario means a 10% drop in income and a 20% increase in debt-to-income ratio, $\delta_s$ would contain these relative changes (e.g., -0.1 for income, +0.2 for DTI).
*   $\odot$: This symbol denotes *element-wise multiplication*. It means that each feature in your original vector $x$ is multiplied by its corresponding $(1 + \delta_s)$ factor. For example, if your income feature is $x_{\text{income}}$ and the income shock is $\delta_{s,\text{income}}$, the stressed income will be $x_{\text{income}} \cdot (1 + \delta_{s,\text{income}})$.

This elegant transformation allows us to simulate the effect of adverse conditions directly on the input features. Instead of manually changing each feature for every scenario, we define a structured shock vector $\delta_s$ that systematically alters the inputs, enabling us to observe the model's behavior under stress.

### Typical Scenario Types

The framework supports a wide range of financially meaningful scenarios:

*   **Macro downturn:** These scenarios reflect broad economic contractions. For example, a macro downturn might involve an increase in unemployment proxy features, a reduction in average income, or a rise in general credit utilization ratios across a portfolio.
*   **Market stress:** These scenarios focus on market-specific shocks. Examples include higher market volatility indices, lower liquidity indicators for certain assets, or a widening of credit spreads.
*   **Idiosyncratic borrower shocks:** These are specific to individual entities or borrowers, rather than systemic. An example could be a personal income drop for a specific customer segment, an increase in delinquency indicators, or a higher Loan-to-Value (LTV) ratio for collateral in a particular lending segment.

By defining a library of such transformations, we can systematically explore the robustness of our financial models against a spectrum of plausible adverse events, which is crucial for risk management and compliance.

## 3. Defining Scenarios and Measuring Model Response
Duration: 0:10:00

Now that we understand the concept of stress transformations, let's actively define a scenario and see its immediate impact on a model's predictions.

<aside class="positive">
<b>Navigation:</b> In the Streamlit application's sidebar, select "<b>Scenario Definition & Model Response</b>".
</aside>

This page provides an interactive panel to define a financial scenario and observe how a hypothetical financial model responds.

### Setup: Hypothetical Model and Data

For demonstration purposes, the application uses a synthetic dataset and a simple Logistic Regression model. Imagine this model is designed to predict the Probability of Default (PD) for individual accounts based on three key features: `income`, `debt_to_income_ratio`, and `utilization_rate`. The model has been "trained" on this synthetic data, and we will now use it to test robustness.

<aside class="positive">
You don't need to interact with the model training part; it's handled automatically to provide a working example.
</aside>

### Define Your Scenario Shocks

This is where you bring your financial expertise into play. Use the sliders provided to define your stress scenario:

*   **Income Scale Factor:** Adjust this to simulate a change in borrower income. A value of 0.9, for instance, means a 10% drop in income.
*   **Debt-to-Income Ratio Scale Factor:** Increase this to simulate a higher debt burden relative to income. A value of 1.2 would mean a 20% increase in DTI.
*   **Utilization Rate Scale Factor:** Adjust this to reflect changes in how much credit is being used. A value of 1.1 might represent a 10% increase in credit card utilization.

As you move the sliders, observe how the "Current scenario shocks ($\delta_s$)" display updates. This array shows the specific percentage changes (or fractional changes) that will be applied to each feature according to your scenario definition.

### Model Response Measurement

After defining your scenario, the application automatically performs two types of predictions for each account in our synthetic portfolio:

*   **Baseline prediction ($\hat{y} = f_\theta(x)$):** This is the model's prediction of Probability of Default (PD) under normal, unstressed conditions, using the original features $x$.
*   **Stressed prediction ($\hat{y}^{(s)} = f_\theta(x^{(s)})$):** This is the model's prediction after the input features $x$ have been transformed into $x^{(s)}$ by applying your defined stress scenario $T_s(x)$.

The difference between these two predictions gives us the impact of the stress:
$$ \Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y} $$
For a credit model, this $\Delta \hat{y}^{(s)}$ represents the change in Probability of Default (PD) for an individual account under stress.

Observe the following output metrics:
*   **Mean Baseline PD:** The average Probability of Default across the entire portfolio before any stress is applied.
*   **Mean Stressed PD:** The average Probability of Default across the portfolio after applying your defined stress scenario.
*   **Mean Change in PD ($\Delta \hat{y}^{(s)}$):** The average difference between stressed and baseline PDs. A positive value indicates that, on average, the portfolio's risk increases under your defined stress.

### Distribution of Change in PD

Below these metrics, you will see a histogram titled "Distribution of Individual Changes in Probability of Default (PD)".

*   **Interpretation:** This plot is crucial for understanding the granularity of the stress impact. It shows how the predicted PD for *each individual account* changes under the stress scenario.
    *   If the histogram is centered around zero, it implies the stress had little impact.
    *   If it shifts significantly to the right (positive $\Delta PD$), it means a widespread increase in PD across many accounts.
    *   The shape and spread of the histogram can reveal whether the impact is uniform or highly variable across the portfolio.

Experiment with different slider values to define various scenarios (e.g., a severe income drop, or a combination of income drop and DTI increase) and observe how the mean PDs and the distribution of $\Delta PD$ change. This hands-on exercise demonstrates the direct and measurable impact of structured stress on your model's outputs.

## 4. Portfolio Aggregation and Sensitivity Analysis
Duration: 0:12:00

In this final step, we move from individual model responses to understanding the aggregated impact across a portfolio and analyzing how the model reacts to varying intensities of stress.

<aside class="positive">
<b>Navigation:</b> In the Streamlit application's sidebar, select "<b>Portfolio Aggregation & Sensitivity</b>".
</aside>

### Portfolio-Level Aggregation

While individual changes in PD are important, risk managers and validators often need to see the aggregated impact across an entire portfolio. This section demonstrates how individual scenario effects are rolled up into meaningful portfolio-level metrics.

Examples of portfolio-level metrics include:
*   Change in mean Probability of Default (PD) or Expected Loss (EL) for a credit portfolio.
*   Shift in factor exposures or Value-at-Risk (VaR) for an equity portfolio.
*   Counts or percentages of obligors whose risk grade upgrades or downgrades under a specific scenario.

**Define Stress Scenario for Aggregation:** Similar to the previous step, use the sliders to define a stress scenario. This scenario will be applied to the entire synthetic portfolio to calculate aggregate metrics.

After setting your preferred stress factors, observe the "Mean Baseline PD (Portfolio)", "Mean Stressed PD (Portfolio)", and "Mean Change in PD (Portfolio)". These metrics provide a high-level summary of the scenario's impact on the overall portfolio risk.

### Risk Grade Migration Under Stress

One critical aspect of portfolio aggregation is understanding how the risk profile of accounts changes. The application demonstrates this using a "Risk Grade Migration Under Stress" table.

*   **Concept:** Imagine we categorize accounts into 'Low Risk', 'Medium Risk', and 'High Risk' based on their PD. This table shows, for example, what percentage of accounts initially in 'Low Risk' under baseline conditions move to 'Medium Risk' or 'High Risk' under the defined stress scenario.
*   **Interpretation:**
    *   The rows represent the **Baseline Risk Grade**.
    *   The columns represent the **Stressed Risk Grade**.
    *   Values on the **diagonal** indicate accounts whose risk grade did not change.
    *   Values **above the diagonal** (e.g., from Low Risk to High Risk) show a downgrade in risk grade, meaning the account became riskier.
    *   Values **below the diagonal** (less common in stress tests, but possible) would indicate an upgrade.
    This table provides a concise overview of how stable the risk grades are under stress.

### Sensitivity and Nonlinearity Analysis

This advanced analysis technique allows us to explore how a model's output evolves as the intensity of a chosen stress scenario gradually increases.

The concept is to introduce a stress intensity parameter, $\alpha$, which scales the initial shock $\delta_s$. The transformed feature vector now becomes:
$$ x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s), \quad \alpha \in [0,1] $$
Here, $\alpha=0$ represents no stress (baseline), and $\alpha=1$ represents the full intensity of your defined stress $\delta_s$. Values between 0 and 1 represent a gradual increase in stress.

**Define a specific scenario for sensitivity analysis:** Use the input boxes provided (e.g., "Income Shock", "DTI Shock", "Utilization Shock") to define the *full* shock (i.e., the $\delta_s$ when $\alpha=1$). For example, entering `-0.1` for Income Shock means a 10% income drop at full stress.

The application then plots the "Mean Stressed PD Trajectory with Increasing Stress Intensity ($\alpha$)".
*   **Interpretation:**
    *   Observe the line as $\alpha$ increases from 0 to 1. This shows how the *average* Probability of Default for the portfolio changes with increasing stress intensity.
    *   If the line is straight, the model's response to stress is linear.
    *   If the line curves or shows sharp increases, it indicates **nonlinearity** in the model's response. This can reveal critical thresholds where the model becomes particularly sensitive, or even unstable, under intensifying stress. Identifying these nonlinearities is vital for understanding model vulnerabilities.

<aside class="negative">
<b>Warning:</b> Nonlinear responses can indicate regions where your model is overly fragile or might produce unexpectedly extreme predictions under severe stress, making it a critical aspect for model validators to scrutinize.
</aside>

This analysis is particularly powerful for generating validation-ready outputs, showing committees not just the impact of a single stress scenario, but the entire trajectory of risk under a range of stress intensities. You have now completed the codelab, gaining a comprehensive understanding of scenario-based robustness testing for ML models.
