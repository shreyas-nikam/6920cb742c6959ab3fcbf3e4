import pytest
import numpy as np
import pandas as pd

# definition_33b4ee3a3a4b4bad90df59197bbf5612 block
from definition_33b4ee3a3a4b4bad90df59197bbf5612 import aggregate_portfolio_impact_metrics
# </your_module> block

@pytest.mark.parametrize(
    "individual_prediction_impacts, portfolio_weights, metric_type, expected",
    [
        # Test case 1: Basic mean aggregation (no weights)
        # Expected functionality: Calculate the simple mean of impacts.
        (pd.Series([10.0, 20.0, 30.0]), None, "mean", 20.0),

        # Test case 2: Weighted mean aggregation
        # Expected functionality: Calculate the weighted mean of impacts.
        (pd.Series([10.0, 20.0, 30.0]), pd.Series([0.1, 0.2, 0.7]), "mean", 26.0),

        # Test case 3: Percentile aggregation (95th percentile, no weights)
        # Expected functionality: Calculate a specific percentile of the impacts.
        # np.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], 95) = 9.55
        (pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]), None, "percentile_95", 9.55),

        # Test case 4: Edge case - Empty individual_prediction_impacts
        # Expected functionality: Handle an empty series gracefully, returning 0.0 for metrics like mean/sum.
        (pd.Series([]), None, "mean", 0.0),

        # Test case 5: Edge case - Invalid metric_type
        # Expected functionality: Raise a ValueError for unsupported aggregation metric types.
        (pd.Series([1.0, 2.0, 3.0]), None, "invalid_metric_type", ValueError),
    ]
)
def test_aggregate_portfolio_impact_metrics(individual_prediction_impacts, portfolio_weights, metric_type, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            aggregate_portfolio_impact_metrics(individual_prediction_impacts, portfolio_weights, metric_type)
    else:
        result = aggregate_portfolio_impact_metrics(individual_prediction_impacts, portfolio_weights, metric_type)
        assert isinstance(result, float)
        # Use pytest.approx for floating-point comparisons to account for precision differences
        assert result == pytest.approx(expected, rel=1e-3, abs=1e-3)