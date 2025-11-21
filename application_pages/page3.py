import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def run_page3():
    st.header("3. Portfolio Aggregation & Sensitivity Analysis")

    # --- Dummy Model and Data Generation (re-used for consistency) ---
    np.random.seed(42)
    n_samples = 1000
    income = np.random.normal(50000, 15000, n_samples)
    debt_to_income_ratio = np.random.beta(2, 5, n_samples) * 0.5 + 0.1
    utilization_rate = np.random.beta(3, 3, n_samples) * 0.8

    prob_default = 1 / (1 + np.exp(-(
        -0.00002 * income +
        5 * debt_to_income_ratio +
        3 * utilization_rate -
        2.5 + np.random.normal(0, 0.5, n_samples)
    )))
    default = (prob_default > np.random.rand(n_samples)).astype(int)

    data = pd.DataFrame({
        'income': income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'utilization_rate': utilization_rate,
        'default': default
    })

    X = data[['income', 'debt_to_income_ratio', 'utilization_rate']]
    y = data['default']

    # For a more stable LR with scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_scaled, y)

    # Baseline predictions
    baseline_predictions = model.predict_proba(X_scaled)[:, 1]

    st.subheader("Portfolio-Level Aggregation")
    st.markdown("""
    Scenario effects can be rolled up from individual predictions to meaningful portfolio-level metrics. This provides a holistic view of the model's robustness across the entire portfolio.

    Examples of portfolio-level metrics:
    *   Change in mean Probability of Default (PD) or Expected Loss (EL) for a credit portfolio.
    *   Shift in factor exposures or Value-at-Risk (VaR) for an equity portfolio.
    *   Counts or percentages of obligors whose risk grade upgrades/downgrades under a specific scenario $s$.

    Let's define a simple scenario and observe its portfolio impact.
    """)

    st.markdown("**Define Stress Scenario for Aggregation:**")
    agg_income_scale = st.slider("Income Scale Factor (for Aggregation)", 0.5, 1.5, 0.9, 0.05)
    agg_dti_scale = st.slider("Debt-to-Income Ratio Scale Factor (for Aggregation)", 0.5, 1.5, 1.2, 0.05)
    agg_util_scale = st.slider("Utilization Rate Scale Factor (for Aggregation)", 0.5, 1.5, 1.1, 0.05)

    delta_s_agg = np.array([agg_income_scale - 1, agg_dti_scale - 1, agg_util_scale - 1])

    # Apply stress transformation to original X, then scale
    X_stressed_values_agg = X.values * (1 + delta_s_agg)
    X_stressed_agg = pd.DataFrame(X_stressed_values_agg, columns=X.columns)
    
    # Ensure non-negative/within bounds
    X_stressed_agg['income'] = X_stressed_agg['income'].clip(lower=0)
    X_stressed_agg['debt_to_income_ratio'] = X_stressed_agg['debt_to_income_ratio'].clip(lower=0, upper=1)
    X_stressed_agg['utilization_rate'] = X_stressed_agg['utilization_rate'].clip(lower=0, upper=1)

    # Apply scaling with the *trained* scaler
    X_stressed_scaled_agg = scaler.transform(X_stressed_agg)

    stressed_predictions_agg = model.predict_proba(X_stressed_scaled_agg)[:, 1]

    df_portfolio = pd.DataFrame({
        'Baseline_PD': baseline_predictions,
        'Stressed_PD': stressed_predictions_agg,
        'PD_Change': stressed_predictions_agg - baseline_predictions
    })

    # Define a simple risk grade system for demonstration
    def get_risk_grade(pd_value):
        if pd_value < 0.05: return 'Low Risk'
        elif pd_value < 0.15: return 'Medium Risk'
        else: return 'High Risk'

    df_portfolio['Baseline_Risk_Grade'] = df_portfolio['Baseline_PD'].apply(get_risk_grade)
    df_portfolio['Stressed_Risk_Grade'] = df_portfolio['Stressed_PD'].apply(get_risk_grade)

    st.write(f"Mean Baseline PD (Portfolio): `{df_portfolio['Baseline_PD'].mean():.4f}`")
    st.write(f"Mean Stressed PD (Portfolio): `{df_portfolio['Stressed_PD'].mean():.4f}`")
    st.write(f"Mean Change in PD (Portfolio): `{df_portfolio['PD_Change'].mean():.4f}`")

    st.subheader("Risk Grade Migration Under Stress")
    grade_migration = pd.crosstab(df_portfolio['Baseline_Risk_Grade'], df_portfolio['Stressed_Risk_Grade'], normalize='index').mul(100).round(2)
    st.dataframe(grade_migration)
    st.markdown("""
    The table above shows the percentage of obligors migrating from their baseline risk grade to a stressed risk grade. Values on the diagonal indicate no change, while off-diagonal values show upgrades (top-right) or downgrades (bottom-left).
    """)

    st.subheader("Sensitivity and Nonlinearity Analysis")
    st.markdown("""
    This analysis explores gradual stress paths by varying the shock intensity $\alpha$ for a chosen scenario. The transformed features are calculated as:

    $$ x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s), \quad \alpha \in [0,1] $$

    By plotting the trajectory of model outputs $\hat{y}^{(s,\alpha)}$ as $\alpha$ increases from 0 to 1, we can reveal nonlinear thresholds and regions where the model's response accelerates or becomes unstable. This helps identify vulnerabilities that might not be apparent from a single stress point.
    """)

    st.markdown("**Define a specific scenario for sensitivity analysis:**")
    sens_income_shock = st.number_input("Income Shock (e.g., -0.1 for 10% drop)", value=-0.1, step=0.01, format="%.2f")
    sens_dti_shock = st.number_input("DTI Shock (e.g., 0.2 for 20% increase)", value=0.2, step=0.01, format="%.2f")
    sens_util_shock = st.number_input("Utilization Shock (e.g., 0.15 for 15% increase)", value=0.15, step=0.01, format="%.2f")

    # Define the base delta_s for the sensitivity analysis
    base_delta_s = np.array([sens_income_shock, sens_dti_shock, sens_util_shock])

    alpha_values = np.linspace(0, 1, 20)
    mean_stressed_pds = []

    for alpha in alpha_values:
        alpha_delta_s = alpha * base_delta_s
        X_alpha_stressed_values = X.values * (1 + alpha_delta_s)
        X_alpha_stressed = pd.DataFrame(X_alpha_stressed_values, columns=X.columns)
        
        # Ensure non-negative/within bounds
        X_alpha_stressed['income'] = X_alpha_stressed['income'].clip(lower=0)
        X_alpha_stressed['debt_to_income_ratio'] = X_alpha_stressed['debt_to_income_ratio'].clip(lower=0, upper=1)
        X_alpha_stressed['utilization_rate'] = X_alpha_stressed['utilization_rate'].clip(lower=0, upper=1)

        # Apply scaling
        X_alpha_stressed_scaled = scaler.transform(X_alpha_stressed)
        
        stressed_predictions_alpha = model.predict_proba(X_alpha_stressed_scaled)[:, 1]
        mean_stressed_pds.append(np.mean(stressed_predictions_alpha))

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=alpha_values, y=mean_stressed_pds, mode='lines+markers', name='Mean Stressed PD'))
    fig_sens.update_layout(title='Mean Stressed PD Trajectory with Increasing Stress Intensity ($\alpha$)',
                           xaxis_title='Stress Intensity ($\alpha$)',
                           yaxis_title='Mean Probability of Default (PD)',
                           hovermode='x unified')
    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown("""
    This plot illustrates how the mean Probability of Default (PD) evolves as the intensity of the defined stress scenario increases. Nonlinearities or sharp increases indicate regions where the model is particularly sensitive or unstable under stress.
    """)
