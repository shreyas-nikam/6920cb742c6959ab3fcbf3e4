import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# definition_9ceeffc0fd6242a186defc8028595446 block
from definition_9ceeffc0fd6242a186defc8028595446 import plot_delta_prediction_distribution
# End definition_9ceeffc0fd6242a186defc8028595446 block

@pytest.mark.parametrize(
    "delta_predictions_series, threshold, expected",
    [
        # Test 1: Valid pd.Series with a float threshold
        # Expected to return a matplotlib.figure.Figure object.
        (pd.Series([1, 2, 3, 0.5, -1, 4], dtype=float), 2.5, plt.Figure),
        
        # Test 2: Valid pd.Series with None threshold (optional argument)
        # Expected to return a matplotlib.figure.Figure object.
        (pd.Series([-0.5, 0.1, 0.9, -0.2], dtype=float), None, plt.Figure),
        
        # Test 3: Empty pd.Series input (edge case)
        # Should gracefully handle empty data and still return a Figure.
        (pd.Series([], dtype=float), 1.0, plt.Figure),
        
        # Test 4: Valid np.ndarray input with a threshold (as specified in docstring)
        # Expected to return a matplotlib.figure.Figure object.
        (np.array([10, -5, 12, 1, -20], dtype=float), 8.0, plt.Figure),
        
        # Test 5: Invalid type for delta_predictions_series (e.g., plain list)
        # Expected to raise a TypeError as it requires pd.Series or np.ndarray.
        ([1, 2, 3], 1.0, TypeError),
    ]
)
def test_plot_delta_prediction_distribution(delta_predictions_series, threshold, expected):
    """
    Tests the plot_delta_prediction_distribution function for various inputs,
    including valid series/arrays, optional threshold, empty series, and type errors.
    """
    if isinstance(expected, type) and issubclass(expected, Exception):
        # If an exception is expected, assert that it is raised.
        with pytest.raises(expected):
            plot_delta_prediction_distribution(delta_predictions_series, threshold)
    else:
        # If a return value type is expected, assert that the function returns an instance of that type.
        fig = plot_delta_prediction_distribution(delta_predictions_series, threshold)
        assert isinstance(fig, expected)
        
        # Close the figure to free up memory and prevent potential warnings
        # when running many tests, especially in non-interactive environments.
        if isinstance(fig, plt.Figure):
            plt.close(fig)