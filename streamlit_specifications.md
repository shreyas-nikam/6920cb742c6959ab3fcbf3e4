
# Streamlit Application Requirements Specification: Scenario-Based Model Robustness Test

## 1. Application Overview

This Streamlit application will provide an interactive platform for users to perform scenario-based robustness tests on machine learning models. It aims to evaluate model stability under various user-defined stress transformations applied to input features.

### Learning Goals
*   Understand the 'Robustness Concept' and the importance of scenario testing for machine learning models, particularly in contexts like finance.
*   Learn to implement and configure stress transformations on specified input columns of a dataset.
*   Evaluate model stability by comparing prediction distributions under stress scenarios against a baseline.
*   Calculate and interpret quantitative metrics such as the mean shift in predictions.
*   Gain insights into model design, monitoring strategies, and constraints based on robustness test results.

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will feature a clear, intuitive layout:
*   **Sidebar:** Will house input controls for model/data upload, scenario configuration, and column selection.
*   **Main Content Area:** Will display application explanations, interactive visualizations, and quantitative results. Sections within the main area will guide the user through the robustness testing process.

### Input Widgets and Controls
*   **Model Upload/Selection:**
    *   A mechanism (e.g., file uploader for a serialized model or an option to use a pre-loaded dummy model) to provide a pre-trained machine learning model with a `.predict` method.
    *   **Annotation:** "Upload your pre-trained model (e.g., `.pkl`, `.joblib`) or use the provided dummy model."
*   **Dataset Upload:**
    *   `st.file_uploader("Upload your test dataset (CSV or Parquet)", type=['csv', 'parquet'])` for `X_test`.
    *   **Annotation:** "This dataset will be used to test the model's robustness under various scenarios."
*   **Volatility Columns Selection:**
    *   `st.multiselect("Select Volatility Columns to Stress", options=list(X_test.columns))` for `vol_cols`.
    *   **Annotation:** "These numerical columns will be subject to stress transformations (e.g., multiplication by a factor)."
*   **Stress Factor Configuration:**
    *   `st.number_input("Factor for 'Volatility Up' Scenario", min_value=1.0, value=2.0, step=0.1, key='factor_up')`
    *   `st.number_input("Factor for 'Volatility Down' Scenario", max_value=1.0, value=0.5, step=0.1, key='factor_down')`
    *   **Annotation:** "Define the multiplication factors to simulate upward and downward volatility shocks."
*   **Custom Scenario Definition (Optional):**
    *   A text area or structured input to allow users to define additional custom stress functions (e.g., Python lambda expressions or function bodies).
    *   **Annotation:** "Define custom stress transformations using Python code. The function should accept `X` (DataFrame) and return `X_stressed`."
*   **Action Button:**
    *   `st.button("Run Robustness Test")` to trigger the analysis.

### Visualization Components
*   **Prediction Distribution Plots:**
    *   For baseline predictions (`y_hat_base`): An interactive histogram or Kernel Density Estimate (KDE) plot.
    *   For each stress scenario (`y_hat_stress`): An interactive overlaid histogram or KDE plot, comparing `y_hat_stress` with `y_hat_base` on the same axes.
    *   **Requirement:** Plots should be generated using libraries like Altair or Plotly for interactivity.
*   **Mean Shift Display:**
    *   Annotated text on each scenario's plot clearly stating the calculated mean shift from the baseline.
    *   Alternatively, a separate bar chart visualizing the mean shift for all active scenarios.
*   **Data Table (Optional):**
    *   `st.dataframe()` to display a sample of the uploaded `X_test` and one or more `X_stressed` DataFrames for inspection.

### Interactive Elements and Feedback Mechanisms
*   **Loading Indicators:** `st.spinner()` will be used while the model is predicting or scenarios are running.
*   **Status Messages:** `st.success()`, `st.error()`, `st.warning()` for user feedback on uploads, errors, and warnings.
*   **Tooltips:** Informative tooltips for all input fields and critical display elements.
*   **Session State:** All user inputs (uploaded data, selected columns, factors, custom scenarios) will be preserved using `st.session_state` to prevent loss of state upon re-runs or interactions.

## 3. Additional Requirements

### Annotation and Tooltip Specifications
*   Clear explanations of the 'Robustness Concept' and 'Key Points' will be rendered using Streamlit's markdown capabilities (`st.markdown`).
*   Tooltips will be provided for all interactive widgets to guide the user on their purpose and expected input.
*   Annotations on charts will highlight key metrics (e.g., mean shift) and scenario names.

### Save the states of the fields properly so that changes are not lost
*   The application will leverage `st.session_state` extensively to ensure that uploaded data, selected volatility columns, defined stress factors, and any custom scenario definitions persist across user interactions and page reloads. This will maintain a consistent user experience.

## 4. Notebook Content and Code Requirements

This section outlines the direct inclusion of markdown and code stubs from the Jupyter Notebook into the Streamlit application.

### Markdown Content

The following markdown content will be integrated into the Streamlit application using `st.markdown()`.

#### Title
`st.title("Case 24: Scenario-Based Model Robustness Test")`

#### Robustness Concept
`st.header("Robustness Concept")`
`st.markdown("""Given a model $\\hat{y} = f_{\\theta}(x)$, we wish to evaluate its stability under stress transformations $T_s$ applied to inputs (e.g., volatility shocks). For each scenario $s$:""")`
`st.latex("x^{(s)} = T_s(x), \\quad \\hat{y}^{(s)} = f_{\\theta}(x^{(s)}).")`
`st.markdown("""We compare distributions of $\\hat{y}^{(s)}$ to baseline.""")`

#### Key Points
`st.header("Key Points")`
`st.markdown("""
*   Identify scenarios where model predictions are unstable or extreme.
*   Feed insights back into model design, constraints, or monitoring.
*   Day 3 Expanded Case Studies: Generative AI, LLMs, RAG, and Agentic AI in Finance
""")`

### Code Stubs and Streamlit Integration

The pseudo-code from the notebook will be translated into Python functions and integrated into the Streamlit application logic.

#### 4.1. `stress_scenario_volatility` Function

This function will apply the stress transformation. It will be called based on user-selected `vol_cols` and `factor`.

```python
import pandas as pd
import streamlit as st
# Placeholder for a dummy model for demonstration
class DummyModel:
    def predict(self, X):
        # A simple dummy prediction, e.g., sum of features
        # Ensure it returns a Series or array for consistency
        return X.sum(axis=1) if not X.empty else pd.Series([])

# @st.cache_data # Cache data if X_test is large and not frequently changing
def stress_scenario_volatility(X: pd.DataFrame, factor: float = 2.0, vol_cols: list = None) -> pd.DataFrame:
    """
    Applies a volatility stress transformation to specified columns of a DataFrame.

    Args:
        X (pd.DataFrame): The input DataFrame.
        factor (float): The multiplication factor for the stress.
        vol_cols (list): List of column names to apply the stress to.

    Returns:
        pd.DataFrame: The stressed DataFrame.
    """
    X_stressed = X.copy()
    if vol_cols:
        for col in vol_cols:
            if col in X_stressed.columns:
                X_stressed[col] *= factor
            else:
                st.warning(f"Column '{col}' not found in the dataset. Skipping stress for this column.")
    return X_stressed
```
*   **Streamlit Usage:** This function will be utilized when defining scenarios and when the "Run Robustness Test" button is clicked. Input parameters (`factor`, `vol_cols`) will come from Streamlit widgets and `st.session_state`.

#### 4.2. Baseline Predictions

The baseline predictions will be calculated once the model and `X_test` are available.

```python
# Assuming 'model' is loaded/available and 'X_test' is in st.session_state
# model = st.session_state.get('model', DummyModel()) # Use dummy if no model uploaded
# X_test = st.session_state.get('X_test')

# if model and X_test is not None and not X_test.empty:
#     with st.spinner("Calculating baseline predictions..."):
#         y_hat_base = model.predict(X_test)
#         st.session_state['y_hat_base'] = pd.Series(y_hat_base)
# else:
#     st.warning("Please upload a model and a test dataset to compute baseline predictions.")
```
*   **Streamlit Usage:** `model` will be either the uploaded model or the `DummyModel`. `X_test` will come from `st.session_state` after user upload. The `y_hat_base` will be stored in `st.session_state`.

#### 4.3. Stress Scenario Definition and Execution

The application will define and execute stress scenarios based on user inputs.

```python
# Assuming 'model', 'X_test', 'vol_cols', 'factor_up', 'factor_down' are in st.session_state
# if st.button("Run Robustness Test"):
#     if 'model' not in st.session_state or 'X_test' not in st.session_state or st.session_state.X_test.empty:
#         st.error("Please upload a model and a test dataset first.")
#     elif not st.session_state.get('vol_cols'):
#         st.error("Please select at least one volatility column to apply stress.")
#     else:
#         model = st.session_state.model
#         X_test = st.session_state.X_test
#         vol_cols = st.session_state.vol_cols
#         y_hat_base = model.predict(X_test) # Recalculate or retrieve from session_state

#         # Define scenarios dynamically based on user inputs
#         scenarios = {
#             "Volatility Up": lambda X: stress_scenario_volatility(X, factor=st.session_state.factor_up, vol_cols=vol_cols),
#             "Volatility Down": lambda X: stress_scenario_volatility(X, factor=st.session_state.factor_down, vol_cols=vol_cols),
#             # Add logic for custom scenarios here if implemented via user input
#         }

#         results = {}
#         for name, T in scenarios.items():
#             with st.spinner(f"Running scenario: {name}..."):
#                 X_stress = T(X_test.copy()) # Pass a copy to avoid modifying original X_test
#                 y_hat_stress = model.predict(X_stress)
#                 results[name] = pd.Series(y_hat_stress)
#         st.session_state['scenario_results'] = results
#         st.session_state['y_hat_base'] = pd.Series(y_hat_base) # Ensure baseline is also stored
#         st.success("Robustness test completed!")
```
*   **Streamlit Usage:** This logic will be triggered by `st.button("Run Robustness Test")`. `scenarios` will be dynamically constructed from user inputs. `results` (containing `y_hat_stress` for each scenario) and `y_hat_base` will be stored in `st.session_state` for visualization and further analysis.

#### 4.4. Analyzing Shifts in Distribution and Visualization

After running the scenarios, the application will analyze and display the results.

```python
import altair as alt
# ... (after scenarios are run and results are in st.session_state)

# if 'scenario_results' in st.session_state and 'y_hat_base' in st.session_state:
#     y_hat_base = st.session_state.y_hat_base
#     scenario_results = st.session_state.scenario_results

#     st.subheader("Analysis of Prediction Shifts")

#     # Display baseline distribution
#     st.markdown("### Baseline Prediction Distribution")
#     base_chart_data = pd.DataFrame({'Prediction': y_hat_base, 'Type': 'Baseline'})
#     base_chart = alt.Chart(base_chart_data).mark_area(opacity=0.7).encode(
#         alt.X('Prediction', bin=alt.Bin(maxbins=50)),
#         alt.Y('count()', stack=None),
#         tooltip=['Prediction', 'count()']
#     ).properties(title='Baseline Predictions').interactive()
#     st.altair_chart(base_chart, use_container_width=True)


#     for name, y_hat_s in scenario_results.items():
#         mean_shift = y_hat_s.mean() - y_hat_base.mean()
#         st.markdown(f"### Scenario: {name}")
#         st.write(f"**Mean Shift:** `{mean_shift:.4f}`")

#         # Create data for overlaid plot
#         chart_data = pd.concat([
#             pd.DataFrame({'Prediction': y_hat_base, 'Type': 'Baseline'}),
#             pd.DataFrame({'Prediction': y_hat_s, 'Type': name})
#         ])

#         # Overlaid Histogram/KDE Plot
#         chart = alt.Chart(chart_data).transform_density(
#             'Prediction',
#             as_=['Prediction', 'density'],
#             groupby=['Type']
#         ).mark_area(opacity=0.6).encode(
#             alt.X('Prediction:Q'),
#             alt.Y('density:Q'),
#             alt.Color('Type:N'),
#             tooltip=['Type', 'Prediction', 'density']
#         ).properties(
#             title=f'Prediction Distribution Comparison: {name} vs Baseline'
#         ).interactive()
#         st.altair_chart(chart, use_container_width=True)

# else:
#     st.info("Run the robustness test to see prediction distributions.")
```
*   **Streamlit Usage:** This code will execute conditionally after `st.session_state['scenario_results']` is populated. `st.subheader()`, `st.markdown()`, `st.write()` will be used for text output. `st.altair_chart()` or similar Streamlit functions for visualization libraries will render the plots.
