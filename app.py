import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab - Scenario-Based Model Robustness Test")
st.divider()
st.markdown("""
In this lab, we delve into scenario-based robustness testing of financial Machine Learning (ML) models. This approach utilizes structured stress transformations, denoted as $T_s(x)$, applied directly to the input features of a model. The core idea is to evaluate how a trained model's predictions, $f_\theta(x^{(s)})$, change when exposed to transparent and parameterized shocks (e.g., macro-economic, market-specific, or borrower-level stresses) without the need for retraining the model under each new scenario.

This methodology is particularly relevant for:
*   **Quantitative analysts and model developers** in credit, market, or trading model development.
*   **Risk managers and validators** performing robustness assessments aligned with standards like SR 11-7.
*   **CFA/FRM/PRM candidates** seeking to bridge stress testing concepts with practical ML implementations.

The fundamental principle involves treating a pre-trained model as an entity subject to various "what-if" scenarios. Each scenario is formally defined as a transformation $T_s$ applied to the model's feature vector.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Stress Transformation Library", "Scenario Definition & Model Response", "Portfolio Aggregation & Sensitivity"])
if page == "Stress Transformation Library":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Scenario Definition & Model Response":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Portfolio Aggregation & Sensitivity":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
