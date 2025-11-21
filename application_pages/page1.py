"""
import streamlit as st
import pandas as pd
import numpy as np

def run_page1():
    st.header("1. Stress Transformation Library & Scenario Definition")

    st.markdown("""
    This page allows you to define and parameterize financial scenarios. Each scenario is implemented as a transformation ($T_s(x)$) applied to the input features ($x$).

    The general form of the transformation is:
    $$
x^{(s)} = T_s(x) = x \odot (1 + \delta_s)
    $$
    where $x^{(s)}$ is the stressed feature vector, $x$ is the baseline feature vector, $\delta_s$ is a vector of scenario shocks, and $\odot$ denotes element-wise multiplication.

    Typical scenario types include:
    *   **Macro downturn:** higher unemployment proxy, lower income, higher utilisation.
    *   **Market stress:** higher volatility, lower liquidity, wider spreads.
    *   **Idiosyncratic borrower shocks:** income drop, increased delinquencies, higher LTV.

    Below, you can define multipliers for various features to simulate different stress scenarios. These multipliers will form the $\delta_s$ vector.
    """)

    st.subheader("Define Scenario Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Income-related Shocks**")
        income_shock = st.slider("Income Multiplier", min_value=0.5, max_value=1.5, value=0.9, step=0.05,
                                 help="e.g., 0.9 for a 10% income drop")
        unemployment_shock = st.slider("Unemployment Rate Multiplier (Proxy)", min_value=0.5, max_value=2.0, value=1.2, step=0.1,
                                       help="e.g., 1.2 for a 20% increase in unemployment proxy")

    with col2:
        st.markdown("**Credit-related Shocks**")
        utilization_shock = st.slider("Utilization Multiplier", min_value=0.8, max_value=1.5, value=1.15, step=0.05,
                                      help="e.g., 1.15 for a 15% increase in credit utilization")
        debt_to_income_shock = st.slider("Debt-to-Income (DTI) Multiplier", min_value=0.8, max_value=1.5, value=1.1, step=0.05,
                                          help="e.g., 1.1 for a 10% increase in DTI")

    with col3:
        st.markdown("**Asset/Market-related Shocks**")
        house_price_shock = st.slider("House Price Multiplier", min_value=0.5, max_value=1.2, value=0.85, step=0.05,
                                      help="e.g., 0.85 for a 15% house price drop")
        credit_spread_shock = st.slider("Credit Spread Additive Shock (bps)", min_value=-50, max_value=300, value=50, step=10,
                                        help="e.g., 50 for a 50 bps increase in credit spread")

    st.divider()
    st.subheader("Current Scenario $\delta_s$ Vector")
    st.markdown("Based on your inputs, the current scenario shock vector (relative to baseline) is:")

    # Display the delta_s values for clarity
    delta_s_data = {
        "Feature": ["Income", "Unemployment Proxy", "Utilization", "Debt-to-Income", "House Price", "Credit Spread"],
        "Multiplier/Shift": [income_shock, unemployment_shock, utilization_shock, debt_to_income_shock, house_price_shock, f"+{credit_spread_shock} bps"],
        "Delta Value (for $1+\delta_s$)": [income_shock - 1, unemployment_shock - 1, utilization_shock - 1, debt_to_income_shock - 1, house_price_shock - 1, credit_spread_shock / 10000] # Assuming credit spread is in bps and converted to a fractional change if applied multiplicatively, here treating as additive.
    }
    delta_s_df = pd.DataFrame(delta_s_data)
    st.dataframe(delta_s_df, hide_index=True)

    st.markdown("""
    **Note on Credit Spread**: For this demonstration, we're showing it as an additive shock in basis points. In a real model, it might be applied differently depending on how credit spreads are represented as a feature.
    """)

    st.session_state["scenario_shocks"] = {
        "income_multiplier": income_shock,
        "unemployment_proxy_multiplier": unemployment_shock,
        "utilization_multiplier": utilization_shock,
        "debt_to_income_multiplier": debt_to_income_shock,
        "house_price_multiplier": house_price_shock,
        "credit_spread_additive_bps": credit_spread_shock
    }

    st.success("Scenario parameters defined successfully. Proceed to 'Model Response & Aggregation' to see their impact.")

"""
