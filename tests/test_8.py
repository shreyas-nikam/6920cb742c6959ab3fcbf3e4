import pytest
import pandas as pd
from definition_5f5aa9c4b19a4e5fb178a158bd135153 import generate_gradual_stress_paths

@pytest.mark.parametrize(
    "original_features_df, base_shock_vector_dict, alpha_steps, expected_output_dfs, expected_exception",
    [
        # Test Case 1: Basic functionality with multiple steps and float values.
        # Checks correct scaling across multiple alpha steps.
        (
            pd.DataFrame({'A': [10.0, 20.0], 'B': [100.0, 200.0]}),
            {'A': 0.1, 'B': -0.2},
            3, # alphas: 0.0, 0.5, 1.0
            [
                pd.DataFrame({'A': [10.0, 20.0], 'B': [100.0, 200.0]}), # alpha = 0.0 (original)
                pd.DataFrame({'A': [10.5, 21.0], 'B': [90.0, 180.0]}),  # alpha = 0.5: A * (1 + 0.5*0.1), B * (1 + 0.5*-0.2)
                pd.DataFrame({'A': [11.0, 22.0], 'B': [80.0, 160.0]})   # alpha = 1.0: A * (1 + 1.0*0.1), B * (1 + 1.0*-0.2)
            ],
            None
        ),
        # Test Case 2: Edge case - alpha_steps = 1.
        # Should result in a list containing only the original DataFrame (alpha=0).
        (
            pd.DataFrame({'X': [1.0, 2.0, 3.0]}),
            {'X': 0.5},
            1,
            [
                pd.DataFrame({'X': [1.0, 2.0, 3.0]})
            ],
            None
        ),
        # Test Case 3: Edge case - Empty original_features_df.
        # Should return a list of empty DataFrames, preserving column structure if any.
        (
            pd.DataFrame(columns=['A', 'B', 'C']), # Empty DF with columns
            {'A': 0.1, 'C': 0.2}, # Shocks are defined but no data to apply them to
            4,
            [
                pd.DataFrame(columns=['A', 'B', 'C']),
                pd.DataFrame(columns=['A', 'B', 'C']),
                pd.DataFrame(columns=['A', 'B', 'C']),
                pd.DataFrame(columns=['A', 'B', 'C'])
            ],
            None
        ),
        # Test Case 4: Edge case - base_shock_vector_dict is empty.
        # No shocks should be applied, so all generated DataFrames should be identical to the original.
        (
            pd.DataFrame({'Feature1': [1.0, 2.0], 'Feature2': [3.0, 4.0]}),
            {}, # Empty shock dictionary
            2, # alphas: 0.0, 1.0
            [
                pd.DataFrame({'Feature1': [1.0, 2.0], 'Feature2': [3.0, 4.0]}),
                pd.DataFrame({'Feature1': [1.0, 2.0], 'Feature2': [3.0, 4.0]})
            ],
            None
        ),
        # Test Case 5: Error handling - Invalid alpha_steps (e.g., 0).
        # The function should raise a ValueError for non-positive alpha_steps.
        (
            pd.DataFrame({'Data': [1.0]}),
            {'Data': 0.1},
            0, # Invalid number of steps
            None,
            ValueError # Expecting a ValueError
        ),
    ]
)
def test_generate_gradual_stress_paths(original_features_df, base_shock_vector_dict, alpha_steps, expected_output_dfs, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            generate_gradual_stress_paths(original_features_df, base_shock_vector_dict, alpha_steps)
    else:
        result_dfs = generate_gradual_stress_paths(original_features_df, base_shock_vector_dict, alpha_steps)

        assert isinstance(result_dfs, list)
        assert len(result_dfs) == alpha_steps

        for i in range(alpha_steps):
            pd.testing.assert_frame_equal(result_dfs[i], expected_output_dfs[i], check_dtype=True)