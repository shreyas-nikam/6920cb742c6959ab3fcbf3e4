"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Helper function to generate dummy data and train a dummy model
@st.cache_data
def get_dummy_model_and_data():
    np.random.seed(42)
    num_samples = 1000
    data = {
        'income': np.random.normal(50000, 15000, num_samples),
        'debt_to_income': np.random.normal(0.3, 0.1, num_samples),
        'utilization': np.random.normal(0.4, 0.2, num_samples),
        'credit_score': np.random.randint(300, 850, num_samples),
        'unemployment_proxy': np.random.normal(0.05, 0.02, num_samples),
        'house_price': np.random.normal(300000, 100000, num_samples),
        'credit_spread': np.random.normal(0.015, 0.005, num_samples) # as a decimal, e.g., 150 bps
    }
    df = pd.DataFrame(data)

    # Ensure some realistic bounds
    df['debt_to_income'] = np.clip(df['debt_to_income'], 0.05, 0.6)
    df['utilization'] = np.clip(df['utilization'], 0.05, 0.9)
    df['unemployment_proxy'] = np.clip(df['unemployment_proxy'], 0.01, 0.15)
    df['credit_spread'] = np.clip(df['credit_spread'], 0.005, 0.035)
    df['income'] = np.clip(df['income'], 20000, 150000)

    # Generate a target variable (e.g., probability of default - PD)
    # Simulating that lower income, higher DTI, higher utilization, lower credit score, higher unemployment, lower house price, higher credit spread lead to higher PD
    df['log_odds'] = (
        -0.00002 * df['income']
        + 5 * df['debt_to_income']
        + 3 * df['utilization']
        - 0.005 * df['credit_score']
        + 50 * df['unemployment_proxy']
        - 0.000002 * df['house_price']
        + 100 * df['credit_spread']
        + np.random.normal(0, 0.5, num_samples)
    )
    df['default'] = (1 / (1 + np.exp(-df['log_odds'])) > 0.5).astype(int)

    X = df[['income', 'debt_to_income', 'utilization', 'credit_score', 'unemployment_proxy', 'house_price', 'credit_spread']]
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test.copy()


def apply_stress_transformation(X_df, scenario_shocks):
    X_stressed = X_df.copy()

    # Apply multiplicative shocks
    if "income_multiplier" in scenario_shocks: 
        X_stressed['income'] = X_stressed['income'] * scenario_shocks['income_multiplier']
    if "unemployment_proxy_multiplier" in scenario_shocks:    
        X_stressed['unemployment_proxy'] = X_stressed['unemployment_proxy'] * scenario_shocks['unemployment_proxy_multiplier']
    if "utilization_multiplier" in scenario_shocks:    
        X_stressed['utilization'] = X_stressed['utilization'] * scenario_shocks['utilization_multiplier']
    if "debt_to_income_multiplier" in scenario_shocks:    
        X_stressed['debt_to_income'] = X_stressed['debt_to_income'] * scenario_shocks['debt_to_income_multiplier']
    if "house_price_multiplier" in scenario_shocks:    
        X_stressed['house_price'] = X_stressed['house_price'] * scenario_shocks['house_price_multiplier']
    
    # Apply additive shock for credit spread (bps to decimal)
    if "credit_spread_additive_bps" in scenario_shocks:    
        X_stressed['credit_spread'] = X_stressed['credit_spread'] + (scenario_shocks['credit_spread_additive_bps'] / 10000)

    # Ensure values remain within realistic bounds after shocking
    X_stressed['debt_to_income'] = np.clip(X_stressed['debt_to_income'], 0.05, 0.6)
    X_stressed['utilization'] = np.clip(X_stressed['utilization'], 0.05, 0.9)
    X_stressed['unemployment_proxy'] = np.clip(X_stressed['unemployment_proxy'], 0.01, 0.15)
    X_stressed['credit_spread'] = np.clip(X_stressed['credit_spread'], 0.005, 0.035)
    X_stressed['income'] = np.clip(X_stressed['income'], 20000, 150000)

    return X_stressed

def run_page2():
    st.header("2. Model Response Measurement & Portfolio-Level Aggregation")

    st.markdown("""
    Here we will measure how a trained model ($f_\theta$) responds to the defined stress scenarios. We will compare baseline predictions with stressed predictions and aggregate the impact at a portfolio level.

    ### Model Response
    *   **Baseline prediction:** $\hat{y} = f_\theta(x)$
    *   **Stressed prediction:** $\hat{y}^{(s)} = f_\theta(x^{(s)})$

    We then compute impact metrics, such as:
    $$
    \Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y}
    $$
    For credit models, this could represent the change in Probability of Default (PD).

    ### Portfolio-Level Aggregation
    The effects are rolled up to portfolio metrics, such as:
    *   Change in mean PD or loss for a credit portfolio.
    *   Counts or percentages of obligors whose risk grade upgrades/downgrades under scenario (s).
    """)

    model, scaler, X_baseline = get_dummy_model_and_data()

    if "scenario_shocks" not in st.session_state:
        st.warning("Please define scenario parameters on the 'Scenario Definition' page first.")
        return

    scenario_shocks = st.session_state["scenario_shocks"]

    st.subheader("Baseline vs. Stressed Predictions")

    # Baseline Predictions
    X_baseline_scaled = scaler.transform(X_baseline)
    baseline_predictions = model.predict_proba(X_baseline_scaled)[:, 1] # Probability of default

    # Stressed Features
    X_stressed = apply_stress_transformation(X_baseline, scenario_shocks)
    X_stressed_scaled = scaler.transform(X_stressed)
    stressed_predictions = model.predict_proba(X_stressed_scaled)[:, 1]

    df_results = X_baseline.copy()
    df_results["Baseline_PD"] = baseline_predictions
    df_results["Stressed_PD"] = stressed_predictions
    df_results["Delta_PD"] = df_results["Stressed_PD"] - df_results["Baseline_PD"]

    st.dataframe(df_results.head(), caption="First 5 individual predictions (Baseline vs. Stressed PD)")

    st.subheader("Portfolio-Level Aggregation")

    mean_baseline_pd = df_results["Baseline_PD"].mean()
    mean_stressed_pd = df_results["Stressed_PD"].mean()
    delta_mean_pd = mean_stressed_pd - mean_baseline_pd

    st.metric(label="Mean Baseline PD", value=f"{mean_baseline_pd:.4f}")
    st.metric(label="Mean Stressed PD", value=f"{mean_stressed_pd:.4f}")
    st.metric(label="Change in Mean PD ($\Delta \hat{y}^{(s)}$)", value=f"{delta_mean_pd:.4f}",
              delta=f"{delta_mean_pd:.4f}")

    st.markdown("""
    ### Distribution of Change in PD ($\Delta PD$)
    This histogram visualizes the distribution of changes in Probability of Default for individual accounts under the defined stress scenario. A shift towards higher positive $\Delta PD$ indicates a widespread increase in risk.
    """)

    fig = go.Figure(data=[go.Histogram(x=df_results['Delta_PD'], nbinsx=50)])
    fig.update_layout(title_text='Distribution of $\Delta PD$ Across Portfolio',
                      xaxis_title_text='Change in PD (Stressed - Baseline)',
                      yaxis_title_text='Number of Accounts')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Risk Grade Migration
    We can also look at how many accounts would 'migrate' to a higher risk category (e.g., higher PD band) under stress.
    """)

    # Define simple risk bands for demonstration
    def get_risk_band(pd_value):
        if pd_value < 0.1:
            return "Low Risk"
        elif pd_value < 0.3:
            return "Medium Risk"
        else:
            return "High Risk"

    df_results['Baseline_Risk_Band'] = df_results['Baseline_PD'].apply(get_risk_band)
    df_results['Stressed_Risk_Band'] = df_results['Stressed_PD'].apply(get_risk_band)

    risk_migration = pd.crosstab(df_results['Baseline_Risk_Band'], df_results['Stressed_Risk_Band'],
                                 normalize='index').round(2)

    st.dataframe(risk_migration, caption="Risk Grade Migration Matrix (Row: Baseline, Column: Stressed)")
    st.info("Numbers represent the proportion of accounts from a baseline risk band that fall into a stressed risk band.")

    st.session_state["df_results_page2"] = df_results
    st.session_state["model_page2"] = model
    st.session_state["scaler_page2"] = scaler
    st.session_state["X_baseline_page2"] = X_baseline


"""
