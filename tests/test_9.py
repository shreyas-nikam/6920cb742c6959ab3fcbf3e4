import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Placeholder for your module import
from definition_7f12ab96d26949a88ece0c3bf0641a3d import plot_model_response_trajectory

def test_plot_model_response_trajectory_valid_input():
    """
    Test case 1: Happy path with valid alpha values, DataFrame, and metric name.
    Expects a matplotlib Figure object and at least one axis.
    """
    alpha_values = np.array([0.1, 0.2, 0.3, 0.4])
    model_output_trajectories_df = pd.DataFrame({
        'MetricA': [1.0, 1.5, 2.0, 2.5],
        'MetricB': [10.0, 11.0, 12.0, 13.0]
    })
    output_metric_name = 'MetricA'

    fig = plot_model_response_trajectory(alpha_values, model_output_trajectories_df, output_metric_name)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0  # Check if at least one subplot was created
    plt.close(fig) # Close the figure to prevent memory leaks

def test_plot_model_response_trajectory_empty_data():
    """
    Test case 2: Empty DataFrame (0 rows, with columns) and empty alpha_values.
    Expects a matplotlib Figure object, which might be an empty plot.
    """
    alpha_values = np.array([])
    model_output_trajectories_df = pd.DataFrame(columns=['MetricA', 'MetricB'])
    output_metric_name = 'MetricA' # Metric name exists as a column, but has no data.

    fig = plot_model_response_trajectory(alpha_values, model_output_trajectories_df, output_metric_name)
    assert isinstance(fig, plt.Figure)
    # Depending on implementation, len(fig.axes) could be 0 or 1 for an empty plot.
    # The key is that it should not raise an error and return a Figure.
    plt.close(fig)

@pytest.mark.parametrize("alpha_values, df, metric_name, expected_exception", [
    # Test case 3: output_metric_name is not present in the DataFrame.
    (np.array([0.1, 0.2]), pd.DataFrame({'MetricA': [1.0, 1.5]}), 'NonExistentMetric', KeyError),
    # Test case 4: output_metric_name is not a string (e.g., None).
    (np.array([0.1, 0.2]), pd.DataFrame({'MetricA': [1.0, 1.5]}), None, TypeError),
    # Test case 5: Length of alpha_values does not match DataFrame rows.
    (np.array([0.1, 0.2, 0.3]), pd.DataFrame({'MetricA': [1.0, 1.5]}), 'MetricA', ValueError),
])
def test_plot_model_response_trajectory_error_cases(alpha_values, df, metric_name, expected_exception):
    """
    Test cases covering various error scenarios using parametrization:
    - Missing output_metric_name in DataFrame (KeyError).
    - Invalid type for output_metric_name (TypeError).
    - Mismatch between alpha_values length and DataFrame rows (ValueError).
    """
    with pytest.raises(expected_exception):
        plot_model_response_trajectory(alpha_values, df, metric_name)