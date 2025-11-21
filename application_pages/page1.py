import streamlit as st

def run_page1():
    st.header("1. Stress Transformation Library ($T_s(x)$)")
    st.markdown("""
    This section introduces the concept of a Stress Transformation Library, which is a family of transformations $T_s: \mathbb{R}^d \to \mathbb{R}^d$ designed to encode different financial scenarios. For a given feature vector $x$ and a specific scenario $s$, the transformed (stressed) feature vector $x^{(s)}$ is calculated as:

    $$ x^{(s)} = T_s(x) = x \odot (1 + \delta_s) $$

    Where:
    *   $x$ is the original feature vector.
    *   $\delta_s$ is a vector of scenario shocks. These shocks quantify the impact of a scenario on each feature (e.g., +20% debt-to-income, –10% income, +200 bps credit spread).
    *   $\odot$ denotes element-wise multiplication.

    This transformation allows us to simulate the effect of adverse conditions directly on the input features, thereby observing the model's behavior under stress.

    ### Typical Scenario Types

    The types of scenarios that can be implemented through these transformations are diverse and can cover various aspects of financial risk:

    *   **Macro downturn:** This might involve shocks like an increase in unemployment proxy features, a reduction in income, or a rise in credit utilization ratios.
    *   **Market stress:** Scenarios could include higher market volatility, lower liquidity indicators, or widening credit spreads.
    *   **Idiosyncratic borrower shocks:** These are specific to individual borrowers, such as a personal income drop, an increase in delinquency indicators, or a higher Loan-to-Value (LTV) ratio for collateral.

    By defining a library of such transformations, we can systematically explore the robustness of our financial models against a spectrum of plausible adverse events.
    """)
