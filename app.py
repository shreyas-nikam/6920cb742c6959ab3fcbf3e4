"""import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we explore scenario-based robustness testing of financial Machine Learning (ML) models. The core idea is to apply structured stress transformations ($T_s(x)$) on input features to observe how a model's predictions ($f_\theta(x^{(s)})$) change, rather than retraining the model for every scenario.

This approach is crucial for quantitative analysts, model developers, risk managers, and validators who need to assess model fragility and ensure compliance with regulations like SR 11-7. It also serves CFA/FRM/PRM candidates by bridging stress testing concepts with ML implementations.

We treat a trained model as an object that is subjected to various 