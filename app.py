"""
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we explore scenario-based robustness testing of financial Machine Learning (ML) models. The core idea is to apply structured stress transformations ($T_s(x)$) on input features to observe how a model's predictions ($f_\theta(x^{(s)})$) change, rather than retraining the model for every scenario.

This approach is crucial for quantitative analysts, model developers, risk managers, and validators who need to assess model fragility and ensure compliance with regulations like SR 11-7. It also serves CFA/FRM/PRM candidates by bridging stress testing concepts with ML implementations.

We treat a trained model as an object that is subjected to various "what-if" scenarios, where each scenario is implemented as a transformation ($T_s$) on the feature vector. This helps us understand how a model behaves under structured, economically meaningful transformations of inputs.

### Learning Outcomes
By completing this case, participants will be able to:

*   Understand how scenario-based stress testing can be applied directly to ML models via feature-level transformations ($T_s(x)$).
*   Translate financial scenarios (e.g., recession, credit spread widening, house price drop) into quantitative shocks on input features.
*   Measure and interpret model sensitivity to structured shocks using risk-friendly metrics (e.g., change in PD, VaR, expected loss, or trading signal).
*   Identify model behaviours that are overly fragile or insufficiently conservative under plausible adverse environments.
*   Integrate scenario-based robustness tests into a model validation and governance workflow.

""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["1. Scenario Definition", "2. Model Response & Aggregation", "3. Sensitivity & Validation"]) 
if page == "1. Scenario Definition":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "2. Model Response & Aggregation":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "3. Sensitivity & Validation":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
"""
