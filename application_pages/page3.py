"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Assuming these helper functions are available or re-defined from page2
# For deployment, consider putting common functions in a utility file
from application_pages.page2 import get_dummy_model_and_data, apply_stress_transformation

def run_page3():
    st.header("3. Sensitivity & Nonlinearity Analysis / Validation-Ready Outputs")

    st.markdown("""
    This section allows for exploring gradual stress paths by varying shock intensity and generating validation-ready outputs.

    ### Sensitivity and Nonlinearity Analysis
    We can vary the shock intensity ($\alpha$) over a range to observe the trajectory of model outputs. The stressed feature vector becomes:
    $$
x^{(s,\alpha)} = x \odot (1 + \alpha \delta_s), \quad \alpha \in [0,1]
    $$
    Plotting $\hat{y}^{(s,\alpha)}$ reveals nonlinear thresholds and regions where the model response accelerates or becomes unstable.

    ### Validation-Ready Outputs
    Generate tables and plots suitable for validation reports and committees:
    *   Scenario vs. baseline metrics
    *   Heatmaps by segment (e.g., rating band, sector, region)
    *   Distributions of ($\Delta \hat{y}^{(s)}$) and flags where changes exceed thresholds
    """)

    if "scenario_shocks" not in st.session_state or \
       "model_page2" not in st.session_state or \
       "scaler_page2" not in st.session_state or \
       "X_baseline_page2" not in st.session_state:
        st.warning("Please define scenario parameters on 'Scenario Definition' and run 'Model Response & Aggregation' first.")
        return

    scenario_shocks = st.session_state["scenario_shocks"]
    model = st.session_state["model_page2"]
    scaler = st.session_state["scaler_page2"]
    X_baseline = st.session_state["X_baseline_page2"]
    df_results_page2 = st.session_state["df_results_page2"]

    st.subheader("Gradual Stress Path Analysis (Varying $\alpha$)")

    alpha_steps = st.slider("Number of $\alpha$ steps (0 to 1)", min_value=5, max_value=50, value=20, step=1)
    alphas = np.linspace(0, 1, alpha_steps)

    portfolio_mean_pds_alpha = []

    # Calculate delta_s for the alpha application
    delta_s_components = {
        "income": scenario_shocks['income_multiplier'] - 1,
        "unemployment_proxy": scenario_shocks['unemployment_proxy_multiplier'] - 1,
        "utilization": scenario_shocks['utilization_multiplier'] - 1,
        "debt_to_income": scenario_shocks['debt_to_income_multiplier'] - 1,
        "house_price": scenario_shocks['house_price_multiplier'] - 1,
        "credit_spread": scenario_shocks['credit_spread_additive_bps'] / 10000
    }

    for alpha in alphas:
        X_alpha_stressed = X_baseline.copy()
        for feature, delta in delta_s_components.items():
            if feature != "credit_spread": # Multiplicative
                X_alpha_stressed[feature] = X_baseline[feature] * (1 + alpha * delta)
            else: # Additive for credit spread
                X_alpha_stressed[feature] = X_baseline[feature] + (alpha * delta)

        # Re-apply clipping to ensure features stay within reasonable bounds after alpha scaling
        X_alpha_stressed['debt_to_income'] = np.clip(X_alpha_stressed['debt_to_income'], 0.05, 0.6)
        X_alpha_stressed['utilization'] = np.clip(X_alpha_stressed['utilization'], 0.05, 0.9)
        X_alpha_stressed['unemployment_proxy'] = np.clip(X_alpha_stressed['unemployment_proxy'], 0.01, 0.15)
        X_alpha_stressed['credit_spread'] = np.clip(X_alpha_stressed['credit_spread'], 0.005, 0.035)
        X_alpha_stressed['income'] = np.clip(X_alpha_stressed['income'], 20000, 150000)

        X_alpha_scaled = scaler.transform(X_alpha_stressed)
        predictions_alpha = model.predict_proba(X_alpha_scaled)[:, 1]
        portfolio_mean_pds_alpha.append(np.mean(predictions_alpha))

    fig_alpha = go.Figure(
        data=[go.Scatter(x=alphas, y=portfolio_mean_pds_alpha, mode='lines+markers', name='Mean Portfolio PD')]
    )
    fig_alpha.update_layout(
        title='Mean Portfolio PD Trajectory Under Gradual Stress (Varying $\alpha$)',
        xaxis_title='Stress Intensity ($\alpha$)',
        yaxis_title='Mean Probability of Default (PD)'
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

    st.markdown("""
    This plot shows how the mean Probability of Default for the entire portfolio changes as the intensity of the defined stress scenario ($\alpha$) increases from 0 (baseline) to 1 (full stress). Deviations from a linear response can indicate nonlinear model behavior or sensitivity thresholds.
    """)

    st.subheader("Validation-Ready Outputs")

    st.markdown("""
    ### Scenario vs. Baseline Metrics Summary
    A direct comparison of key metrics under baseline and the fully stressed scenario.
    """)

    summary_metrics = pd.DataFrame({
        "Metric": ["Mean Probability of Default", "Standard Deviation of PD"],
        "Baseline": [df_results_page2["Baseline_PD"].mean(), df_results_page2["Baseline_PD"].std()],
        "Stressed": [df_results_page2["Stressed_PD"].mean(), df_results_page2["Stressed_PD"].std()],
        "Delta": [df_results_page2["Delta_PD"].mean(), df_results_page2["Delta_PD"].std()]
    }).set_index("Metric")

    st.dataframe(summary_metrics.style.format("{:.4f}"))

    st.markdown("""
    ### Distribution of $\Delta PD$ with Threshold Flagging
    We can identify accounts where the change in PD exceeds a certain threshold, indicating significant impact.
    """)

    delta_pd_threshold = st.slider("Threshold for $\Delta PD$ flagging", min_value=0.0, max_value=0.1, value=0.03, step=0.005)
    df_results_page2["Exceeds_Threshold"] = df_results_page2["Delta_PD"] > delta_pd_threshold

    num_exceeding = df_results_page2["Exceeds_Threshold"].sum()
    percent_exceeding = (num_exceeding / len(df_results_page2)) * 100

    st.info(f"**{num_exceeding}** accounts ({percent_exceeding:.2f}%) have a $\Delta PD$ exceeding the threshold of **{delta_pd_threshold:.3f}**.")

    fig_flagged = go.Figure()
    fig_flagged.add_trace(go.Histogram(x=df_results_page2['Delta_PD'], nbinsx=50,
                                     name='All Accounts', opacity=0.7))
    fig_flagged.add_trace(go.Histogram(x=df_results_page2[df_results_page2['Exceeds_Threshold']]['Delta_PD'],
                                     nbinsx=50, name=f'$\Delta PD$ > {delta_pd_threshold}',
                                     marker_color='red', opacity=0.7))

    fig_flagged.update_layout(barmode='overlay', title_text='Distribution of $\Delta PD$ with Flagged Accounts',
                              xaxis_title_text='Change in PD (Stressed - Baseline)',
                              yaxis_title_text='Number of Accounts')
    st.plotly_chart(fig_flagged, use_container_width=True)

    st.markdown("""
    ### Heatmap by Risk Band (Example: Baseline Risk vs. $\Delta PD$)
    This heatmap helps visualize the average $\Delta PD$ for different segments, based on their baseline risk characteristics.
    """)

    # Re-using Risk Bands from Page 2 if available, otherwise defining again
    def get_risk_band(pd_value):
        if pd_value < 0.1:
            return "Low Risk"
        elif pd_value < 0.3:
            return "Medium Risk"
        else:
            return "High Risk"
    
    if 'Baseline_Risk_Band' not in df_results_page2.columns:
        df_results_page2['Baseline_Risk_Band'] = df_results_page2['Baseline_PD'].apply(get_risk_band)

    pivot_table = df_results_page2.pivot_table(values='Delta_PD', index='Baseline_Risk_Band', aggfunc='mean')

    fig_heatmap = go.Figure(data=go.Heatmap(z=pivot_table.values,
                                           x=pivot_table.columns,
                                           y=pivot_table.index,
                                           colorscale='RdYlGn_r')) # Red-Yellow-Green, reversed for higher delta = more red

    fig_heatmap.update_layout(title_text='Average $\Delta PD$ by Baseline Risk Band',
                              xaxis_title_text='Segment (e.g., placeholder)',
                              yaxis_title_text='Baseline Risk Band',
                              autosize=False, width=700, height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.success("Validation-ready outputs generated successfully.")

"""
