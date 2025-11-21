import pytest
import pandas as pd
from definition_c27f2ed81b124814b45aa7a9f4558b24 import generate_scenario_comparison_report_table

@pytest.mark.parametrize(
    "baseline_metrics, stressed_metrics, expected_output",
    [
        # Test Case 1: Standard comparison with common metrics
        (
            {'PD': 0.01, 'EL': 1000, 'VaR': 5000},
            {'PD': 0.02, 'EL': 1500, 'VaR': 7500},
            pd.DataFrame(
                {'Baseline': [1000.0, 0.01, 5000.0], 'Stressed': [1500.0, 0.02, 7500.0]},
                index=pd.Index(['EL', 'PD', 'VaR'], name='Metric')
            )
        ),
        # Test Case 2: Empty input dictionaries (edge case)
        (
            {},
            {},
            pd.DataFrame(columns=['Baseline', 'Stressed'], index=pd.Index([], name='Metric'))
        ),
        # Test Case 3: Mismatched keys in input dictionaries (some metrics only in one scenario)
        (
            {'PD': 0.01, 'EL': 1000},
            {'PD': 0.02, 'VaR': 7500},
            pd.DataFrame(
                {'Baseline': [1000.0, 0.01, float('nan')], 'Stressed': [float('nan'), 0.02, 7500.0]},
                index=pd.Index(['EL', 'PD', 'VaR'], name='Metric')
            )
        ),
        # Test Case 4: Invalid input type for baseline_metrics (e.g., list instead of dict)
        (
            [1, 2, 3],
            {'PD': 0.02},
            TypeError
        ),
        # Test Case 5: Invalid input type for stressed_metrics (e.g., None instead of dict)
        (
            {'PD': 0.01},
            None,
            TypeError
        ),
    ]
)
def test_generate_scenario_comparison_report_table(baseline_metrics, stressed_metrics, expected_output):
    # If the expected_output is an exception type, we expect the function to raise it
    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        with pytest.raises(expected_output):
            generate_scenario_comparison_report_table(baseline_metrics, stressed_metrics)
    # Otherwise, we expect a DataFrame and compare it using pandas testing utility
    else:
        result_df = generate_scenario_comparison_report_table(baseline_metrics, stressed_metrics)
        pd.testing.assert_frame_equal(result_df, expected_output, check_dtype=True)

