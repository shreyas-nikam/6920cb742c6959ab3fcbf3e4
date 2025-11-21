import pytest
from definition_e9985fd13e3e4778848124cf5c0dfbc5 import create_financial_scenario_shock_vector

@pytest.mark.parametrize("scenario_name, scenario_params_dict, expected", [
    # Test case 1: Standard scenario with multiple parameters
    ("macro_downturn", {"income_scale": 0.9, "utilization_increase": 0.2, "house_price_scale": 0.85}, 
     {"income_scale": 0.9, "utilization_increase": 0.2, "house_price_scale": 0.85}),
    
    # Test case 2: Empty scenario parameters (edge case)
    ("no_shock_scenario", {}, {}),
    
    # Test case 3: Scenario with a single parameter
    ("market_stress_interest_rate", {"interest_rate_shift": 0.015}, 
     {"interest_rate_shift": 0.015}),
    
    # Test case 4: Invalid type for scenario_name (expecting str)
    (123, {"income_scale": 0.9}, TypeError),
    
    # Test case 5: Invalid type for scenario_params_dict (expecting dict)
    ("invalid_params_type", "not_a_dict", TypeError),
])
def test_create_financial_scenario_shock_vector(scenario_name, scenario_params_dict, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            create_financial_scenario_shock_vector(scenario_name, scenario_params_dict)
    else:
        result = create_financial_scenario_shock_vector(scenario_name, scenario_params_dict)
        assert result == expected

