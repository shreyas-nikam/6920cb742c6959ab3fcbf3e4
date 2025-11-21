import pytest
import pandas as pd
# Keep a placeholder definition_b6985ad5fb444542a762008d34873227 for the import of the module. Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_b6985ad5fb444542a762008d34873227 import apply_scenario_shock_to_features

@pytest.mark.parametrize("input_data, expected", [
    # Test Case 1: Basic functionality - apply shocks to a subset of features
    (
        (pd.DataFrame({'income': [1000, 2000, 1500], 'utilization': [0.5, 0.7, 0.6], 'age': [30, 40, 35]}),
         {'income': 0.9, 'utilization': 1.2}),
        pd.DataFrame({'income': [900.0, 1800.0, 1350.0], 'utilization': [0.6, 0.84, 0.72], 'age': [30, 40, 35]})
    ),
    # Test Case 2: Edge Case: Empty shock dictionary - no shocks applied, should return a copy
    (
        (pd.DataFrame({'income': [1000, 2000], 'age': [30, 40]}),
         {}),
        pd.DataFrame({'income': [1000, 2000], 'age': [30, 40]})
    ),
    # Test Case 3: Edge Case: Shocking non-existent features - should ignore unknown features
    (
        (pd.DataFrame({'income': [1000, 2000], 'age': [30, 40]}),
         {'income': 0.9, 'non_existent_feature': 1.1}),
        pd.DataFrame({'income': [900.0, 1800.0], 'age': [30, 40]})
    ),
    # Test Case 4: Error Handling: Invalid features_df type (e.g., None)
    (
        (None, {'income': 0.9}),
        TypeError
    ),
    # Test Case 5: Error Handling: Invalid shock_vector_dict type (e.g., None)
    (
        (pd.DataFrame({'a': [1]}), None),
        TypeError
    )
])
def test_apply_scenario_shock_to_features(input_data, expected):
    features_df, shock_vector_dict = input_data
    
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            apply_scenario_shock_to_features(features_df, shock_vector_dict)
    else:
        result_df = apply_scenario_shock_to_features(features_df, shock_vector_dict)
        pd.testing.assert_frame_equal(result_df, expected)
        
        # Ensure that if the shock_vector_dict is empty, a new DataFrame object is returned (good practice for immutability)
        if features_df is not None and not shock_vector_dict:
            assert id(result_df) != id(features_df)