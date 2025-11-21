import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def run_page2():
    st.header("2. Scenario Definition Panel & Model Response Measurement")

    st.markdown("""
    This section allows users to define and parameterize various financial scenarios by applying multipliers and shifts to selected input features. We will then observe how a hypothetical financial model responds to these defined stresses.

    ### Scenario Definition Panel

    Define your scenario by adjusting the scaling factors for key financial features. These scales will be translated into the $\delta_s$ vector for our stress transformation $T_s(x)$. A value of 1.0 means no change, while values less than 1.0 represent a decrease and values greater than 1.0 represent an increase.
    """)

    # --- Dummy Model and Data Generation ---
    st.subheader("Setup: Hypothetical Model and Data")
    st.markdown("""
    To demonstrate, we will use a synthetic dataset and a simple Logistic Regression model.
    Imagine we have a model predicting Probability of Default (PD) based on `income`, `debt_to_income_ratio`, and `utilization_rate`.
    """)

    np.random.seed(42)
    n_samples = 1000
    income = np.random.normal(50000, 15000, n_samples)
    debt_to_income_ratio = np.random.beta(
        2, 5, n_samples) * 0.5 + 0.1  # Range 0.1 to 0.6
    utilization_rate = np.random.beta(3, 3, n_samples) * 0.8  # Range 0 to 0.8

    # Simulate a target variable (e.g., default) based on features
    # Higher DTI and utilization, lower income -> higher probability of default
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

    # Scale features for the model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a simple Logistic Regression model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_scaled, y)

    st.write("Model trained on synthetic data.")

    # --- Scenario Definition ---
    st.subheader("Define Your Scenario Shocks")
    col1, col2, col3 = st.columns(3)
    with col1:
        income_scale = st.slider("Income Scale Factor", 0.5, 1.5, 1.0, 0.05)
    with col2:
        debt_to_income_scale = st.slider(
            "Debt-to-Income Ratio Scale Factor", 0.5, 1.5, 1.0, 0.05)
    with col3:
        utilization_scale = st.slider(
            "Utilization Rate Scale Factor", 0.5, 1.5, 1.0, 0.05)

    # Calculate delta_s based on scales
    delta_s = np.array(
        [income_scale - 1, debt_to_income_scale - 1, utilization_scale - 1])
    st.markdown(
        f"Current scenario shocks ($\delta_s$): `{delta_s[0]:.2f}, {delta_s[1]:.2f}, {delta_s[2]:.2f}`")

    # --- Model Response Measurement ---
    st.subheader("Model Response Measurement")
    st.markdown("""
    For a trained model $f_\theta$, we compute both a baseline prediction and a stressed prediction:

    *   **Baseline prediction:**
        $$ \hat{y} = f_\theta(x) $$
        This is the model's prediction under normal, unstressed conditions.

    *   **Stressed prediction:**
        $$ \hat{y}^{(s)} = f_\theta(x^{(s)}) $$
        This is the model's prediction when the input features $x$ are transformed by the scenario $T_s(x)$ to $x^{(s)}$.

    We then compute impact metrics, such as the change in prediction:
    $$ \Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y} $$
    Or, for aggregated measures like Expected Loss (EL):
    $$ \Delta \text{EL}^{(s)} = \text{EL}^{(s)} - \text{EL} $$
    """)

    # Perform predictions
    baseline_predictions = model.predict_proba(X_scaled)[:, 1]

    # Apply stress transformation to original X, then scale
    X_stressed_values = X.values * (1 + delta_s)
    X_stressed = pd.DataFrame(X_stressed_values, columns=X.columns)

    # Ensure no negative values for features that should be non-negative
    X_stressed['income'] = X_stressed['income'].clip(lower=0)
    X_stressed['debt_to_income_ratio'] = X_stressed['debt_to_income_ratio'].clip(
        lower=0, upper=1)
    X_stressed['utilization_rate'] = X_stressed['utilization_rate'].clip(
        lower=0, upper=1)

    # Scale the stressed features using the *trained* scaler
    X_stressed_scaled = scaler.transform(X_stressed)

    stressed_predictions = model.predict_proba(X_stressed_scaled)[:, 1]

    delta_predictions = stressed_predictions - baseline_predictions

    st.write(f"Mean Baseline PD: `{np.mean(baseline_predictions):.4f}`")
    st.write(f"Mean Stressed PD: `{np.mean(stressed_predictions):.4f}`")
    st.write(
        "Mean Change in PD ($\Delta \hat{y}^{(s)}$):", f"`{np.mean(delta_predictions):.4f}`")

    # Plotting the distribution of Delta_y
    st.subheader("Distribution of Change in PD")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=delta_predictions, nbinsx=50,
                  name='$\Delta \hat{y}^{(s)}$ (Change in PD)'))
    fig.update_layout(title='Distribution of Individual Changes in Probability of Default (PD)',
                      xaxis_title='Change in PD (Stressed PD - Baseline PD)',
                      yaxis_title='Number of Instances',
                      hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    This histogram shows the distribution of how each individual's predicted Probability of Default (PD) changes under the defined stress scenario. A shift to the right indicates a general increase in PD under stress.
    """)
