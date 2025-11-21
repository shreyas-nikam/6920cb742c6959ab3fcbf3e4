import pytest
import pandas as pd
from definition_6ae6d4691ac74c00b0315a5a86f95a0b import identify_risk_grade_migrations

@pytest.mark.parametrize("baseline_risk_grades_param, stressed_risk_grades_param, expected", [
    # Test Case 1: Standard scenario with mixed migrations (upgrades, downgrades, stable).
    # Assuming lower numeric values represent better risk grades (e.g., 1=best, 5=worst).
    (pd.Series([1, 2, 3, 4, 5], index=['ObligorA', 'ObligorB', 'ObligorC', 'ObligorD', 'ObligorE']),
     pd.Series([2, 1, 3, 5, 4], index=['ObligorA', 'ObligorB', 'ObligorC', 'ObligorD', 'ObligorE']),
     {'upgrades': 2, 'downgrades': 2, 'stable': 1}),

    # Test Case 2: All obligors have stable risk grades.
    (pd.Series([1, 2, 3], index=['X', 'Y', 'Z']),
     pd.Series([1, 2, 3], index=['X', 'Y', 'Z']),
     {'upgrades': 0, 'downgrades': 0, 'stable': 3}),

    # Test Case 3: All obligors experience an upgrade (e.g., from worse to better grade).
    (pd.Series([4, 3, 2], index=[101, 102, 103]),
     pd.Series([3, 2, 1], index=[101, 102, 103]),
     {'upgrades': 3, 'downgrades': 0, 'stable': 0}),

    # Test Case 4: Empty Series for both baseline and stressed grades (edge case).
    (pd.Series([], dtype=int),
     pd.Series([], dtype=int),
     {'upgrades': 0, 'downgrades': 0, 'stable': 0}),

    # Test Case 5: Invalid input type (non-pd.Series for baseline_risk_grades).
    # Expecting a TypeError or AttributeError when Series methods are called on a non-Series object.
    ("not a pandas Series", pd.Series([1, 2]), TypeError),
])
def test_identify_risk_grade_migrations(baseline_risk_grades_param, stressed_risk_grades_param, expected):
    try:
        result = identify_risk_grade_migrations(baseline_risk_grades_param, stressed_risk_grades_param)
        assert result == expected
    except Exception as e:
        assert isinstance(e, expected)
