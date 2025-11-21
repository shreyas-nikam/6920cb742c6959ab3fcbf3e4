import pickle
import seaborn as sns
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_scenario_shock_to_features(features_df, shock_vector_dict):
    """
    Applies a scenario-specific shock transformation to a DataFrame of input features.

    The shock vector contains multipliers for specific features, applied element-wise
    to generate stressed features.

    Arguments:
        features_df (pd.DataFrame): The original DataFrame of input features.
        shock_vector_dict (dict): A dictionary where keys are feature names and values
                                  are the shock multipliers (e.g., 0.9 for -10% income).

    Returns:
        pd.DataFrame: A new DataFrame with features transformed by the scenario shock.

    Raises:
        TypeError: If features_df is not a pandas DataFrame or shock_vector_dict is not a dict.
    """
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df must be a pandas DataFrame.")
    if not isinstance(shock_vector_dict, dict):
        raise TypeError("shock_vector_dict must be a dictionary.")

    stressed_features_df = features_df.copy()

    for feature, shock_multiplier in shock_vector_dict.items():
        if feature in stressed_features_df.columns:
            stressed_features_df[feature] = stressed_features_df[feature] * \
                shock_multiplier

    return stressed_features_df


def create_financial_scenario_shock_vector(scenario_name, scenario_params_dict):
    """Creates a dictionary representing feature-level shocks for a financial scenario.

    Arguments:
    scenario_name (str): The name of the financial scenario.
    scenario_params_dict (dict): Parameters defining the scenario's impact.

    Output:
    dict: A dictionary mapping feature names to their shock multipliers/shifts.
    """
    if not isinstance(scenario_name, str):
        raise TypeError("scenario_name must be a string.")
    if not isinstance(scenario_params_dict, dict):
        raise TypeError("scenario_params_dict must be a dictionary.")

    # In this implementation, the function directly uses the provided
    # scenario_params_dict as the shock vector, implying that the input
    # dictionary already contains the desired feature-level shocks.
    return scenario_params_dict


def load_financial_model(model_path):
    """
    Loads a pre-trained machine learning model from a given file path using pickle.

    Arguments:
        model_path (str): The file path to the saved model (e.g., a pickled scikit-learn model).
    Output:
        Any: The loaded machine learning model object.
    """
    if not isinstance(model_path, str):
        raise TypeError("model_path must be a string.")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        # Handles non-existent paths and empty string paths.
        raise
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
        # These exceptions commonly indicate a corrupted or invalid model file.
        # Re-raising as OSError to match the test case expectation for corrupted files.
        raise OSError(
            f"Failed to load model from '{model_path}': Corrupted file or invalid format. Original error: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during file operations or loading.
        # If it's already an OSError (e.g., permissions issue), re-raise it directly.
        if isinstance(e, OSError):
            raise
        raise OSError(
            f"An unexpected error occurred while loading model from '{model_path}': {e}") from e


def get_baseline_model_predictions(model, original_features_df):
    """
    Calculates the baseline predictions (e.g., probabilities of default, expected loss)
    from the given machine learning model using the original, unstressed input features.

    Arguments:
        model (Any): The pre-trained machine learning model.
        original_features_df (pd.DataFrame): The original, unstressed DataFrame of input features.

    Output:
        pd.Series or np.ndarray: The array of baseline predictions.
    """
    if not isinstance(original_features_df, pd.DataFrame):
        raise TypeError("original_features_df must be a pandas DataFrame.")

    if not hasattr(model, 'predict'):
        raise AttributeError(
            "The provided model object must have a 'predict' method.")

    try:
        predictions = model.predict(original_features_df)
    except Exception as e:
        # Propagate any exceptions that occur during the model's prediction
        raise e

    # The docstring specifies pd.Series or np.ndarray.
    # Most ML models' predict methods return a numpy array.
    # The tests also expect either type. We return the direct output of the model.
    return predictions


def get_stressed_model_predictions(model, stressed_features_df):
    """Calculates the stressed predictions from the given machine learning model using the
    scenario-transformed input features. These predictions reflect the model's output
    under the defined stress conditions.

    Arguments:
        model (Any): The pre-trained machine learning model, expected to have a .predict method.
        stressed_features_df (pd.DataFrame): The DataFrame of input features after applying
                                             the scenario shock transformation.

    Output:
        pd.Series or np.ndarray: The array of stressed predictions.
    """

    # Validate that stressed_features_df is a pandas DataFrame.
    # This explicitly handles cases like Test Case 5, where a non-DataFrame input
    # should raise a TypeError.
    if not isinstance(stressed_features_df, pd.DataFrame):
        raise TypeError("stressed_features_df must be a pandas DataFrame.")

    # Call the model's predict method with the stressed features.
    # This will raise an AttributeError if the model object does not have a 'predict' method,
    # handling cases like Test Case 4.
    predictions = model.predict(stressed_features_df)

    return predictions


def calculate_individual_prediction_impact(baseline_predictions, stressed_predictions):
    """
    Determines the absolute change in predictions for each individual data point
    by comparing stressed predictions against baseline predictions.

    Arguments:
    baseline_predictions (pd.Series or np.ndarray): The array of baseline predictions.
    stressed_predictions (pd.Series or np.ndarray): The array of stressed predictions.

    Output:
    pd.Series or np.ndarray: The array representing the change in predictions (stressed - baseline).
    """
    # Validate input types
    if not isinstance(baseline_predictions, (pd.Series, np.ndarray)) or \
       not isinstance(stressed_predictions, (pd.Series, np.ndarray)):
        raise TypeError(
            "Input predictions must be pandas Series or numpy arrays.")

    # Validate lengths
    if len(baseline_predictions) != len(stressed_predictions):
        raise ValueError(
            "Baseline and stressed predictions must have the same length.")

    # Calculate the difference directly.
    # This operation handles both pandas Series and numpy arrays
    # and preserves the input type if both inputs are of the same type.
    impact = stressed_predictions - baseline_predictions

    return impact


def aggregate_portfolio_impact_metrics(individual_prediction_impacts, portfolio_weights, metric_type):
    """
    Aggregates individual prediction impacts to derive portfolio-level metrics.

    Arguments:
        individual_prediction_impacts (pd.Series or np.ndarray): Changes in predictions.
        portfolio_weights (pd.Series or np.ndarray, optional): Weights for each observation.
        metric_type (str): Type of aggregation metric (e.g., "mean", "percentile_95").

    Output:
        float: The aggregated portfolio-level impact metric.

    Raises:
        ValueError: If weights size does not match impacts, invalid percentile format,
                    or unsupported metric type.
    """
    impacts = np.asarray(individual_prediction_impacts)

    # Handle empty impacts array gracefully
    if impacts.size == 0:
        return 0.0

    weights = None
    if portfolio_weights is not None:
        weights = np.asarray(portfolio_weights)
        if weights.size != impacts.size:
            raise ValueError(
                "portfolio_weights must have the same size as individual_prediction_impacts.")

    if metric_type == "mean":
        if weights is not None:
            # np.average handles unnormalized weights by normalizing them internally
            return np.average(impacts, weights=weights)
        else:
            return np.mean(impacts)
    elif metric_type.startswith("percentile_"):
        try:
            percentile_value = float(metric_type.split("_")[1])
            if not (0 <= percentile_value <= 100):
                raise ValueError("Percentile value must be between 0 and 100.")
            # np.percentile does not support direct weighting for its calculation
            return np.percentile(impacts, percentile_value)
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid percentile metric type: {metric_type}. Expected format 'percentile_XX'.") from e
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")


def identify_risk_grade_migrations(baseline_risk_grades, stressed_risk_grades):
    """
    Compares baseline and stressed risk grades for common obligors to identify
    and quantify risk grade migrations (upgrades, downgrades, stable).
    Lower numeric values indicate better risk grades.

    Args:
        baseline_risk_grades (pd.Series): Original risk grades.
        stressed_risk_grades (pd.Series): Risk grades under stress.

    Returns:
        dict: Counts of upgrades, downgrades, and stable risk grades.

    Raises:
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(baseline_risk_grades, pd.Series) or not isinstance(stressed_risk_grades, pd.Series):
        raise TypeError(
            "Both baseline_risk_grades and stressed_risk_grades must be pandas Series.")

    # Find common obligors to ensure a fair comparison across the same set of entities.
    common_obligors = baseline_risk_grades.index.intersection(
        stressed_risk_grades.index)

    # If there are no common obligors, no migrations can be identified.
    if common_obligors.empty:
        return {'upgrades': 0, 'downgrades': 0, 'stable': 0}

    # Align series to include only the common obligors.
    baseline_aligned = baseline_risk_grades.loc[common_obligors]
    stressed_aligned = stressed_risk_grades.loc[common_obligors]

    # Calculate differences: stressed_grade - baseline_grade.
    # A negative difference indicates an upgrade (stressed grade is numerically lower/better).
    # A positive difference indicates a downgrade (stressed grade is numerically higher/worse).
    # A zero difference indicates a stable grade.
    differences = stressed_aligned - baseline_aligned

    upgrades = (differences < 0).sum()
    downgrades = (differences > 0).sum()
    stable = (differences == 0).sum()

    return {
        'upgrades': int(upgrades),
        'downgrades': int(downgrades),
        'stable': int(stable)
    }


def generate_gradual_stress_paths(original_features_df, base_shock_vector_dict, alpha_steps):
    """    Generates a list of DataFrames, each representing input features stressed at increasing intensity levels (alpha). This allows for simulating a gradual stress path to observe non-linear model responses.
Arguments:
original_features_df (pd.DataFrame): The original, unstressed DataFrame of input features.
base_shock_vector_dict (dict): The dictionary of full shock multipliers for the scenario at alpha=1.
alpha_steps (int): The number of steps between 0 and 1 for the alpha intensity, inclusive. Defaults to 11 (0.0, 0.1, ..., 1.0).
Output:
list[pd.DataFrame]: A list of DataFrames, where each DataFrame corresponds to a specific alpha intensity level.
    """

    if not isinstance(alpha_steps, int) or alpha_steps <= 0:
        raise ValueError("alpha_steps must be a positive integer.")

    stressed_dfs = []

    # Generate alpha values from 0 to 1, inclusive, with alpha_steps number of points.
    # For example, if alpha_steps=3, alphas will be [0.0, 0.5, 1.0]
    # If alpha_steps=1, alphas will be [0.0]
    alphas = np.linspace(0, 1, alpha_steps)

    for alpha in alphas:
        # Create a deep copy of the original DataFrame to apply shocks
        # This ensures the original DataFrame is not modified and each stressed
        # DataFrame is independent.
        current_stressed_df = original_features_df.copy(deep=True)

        # Apply shocks based on the current alpha level
        for feature, shock_multiplier in base_shock_vector_dict.items():
            # Only apply shock if the feature (column) exists in the DataFrame
            if feature in current_stressed_df.columns:
                # The stressed value is calculated as: original_value * (1 + alpha * shock_multiplier)
                current_stressed_df[feature] = current_stressed_df[feature] * \
                    (1 + alpha * shock_multiplier)

        stressed_dfs.append(current_stressed_df)

    return stressed_dfs


def plot_model_response_trajectory(alpha_values, model_output_trajectories_df, output_metric_name):
    """
    Visualizes how a model's predictions or aggregated metrics change as the intensity of a stress scenario gradually increases.
    """
    # Input Validation
    if not isinstance(output_metric_name, str):
        raise TypeError("output_metric_name must be a string.")

    if output_metric_name not in model_output_trajectories_df.columns:
        raise KeyError(
            f"Output metric '{output_metric_name}' not found in DataFrame columns.")

    if len(alpha_values) != len(model_output_trajectories_df):
        raise ValueError(
            "Length of 'alpha_values' must match the number of rows in 'model_output_trajectories_df'.")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data for plotting
    y_data = model_output_trajectories_df[output_metric_name]

    # Plot the trajectory
    ax.plot(alpha_values, y_data, marker='o',
            linestyle='-', label=output_metric_name)

    # Add plot enhancements
    ax.set_xlabel('Stress Intensity (Alpha)')
    ax.set_ylabel(output_metric_name)
    ax.set_title(f'Model Response Trajectory: {output_metric_name} vs. Alpha')
    ax.grid(True)
    ax.legend()

    # Ensure tight layout for better appearance
    fig.tight_layout()

    return fig


def generate_scenario_comparison_report_table(baseline_metrics_dict, stressed_metrics_dict):
    """
    Generates a tabular report comparing key portfolio performance and risk metrics under baseline conditions versus a specified stress scenario.

    Arguments:
    baseline_metrics_dict (dict): A dictionary of aggregated metrics for the baseline scenario.
    stressed_metrics_dict (dict): A dictionary of aggregated metrics for the stressed scenario.

    Output:
    pd.DataFrame: A DataFrame presenting a side-by-side comparison of metrics.
    """

    if not isinstance(baseline_metrics_dict, dict):
        raise TypeError("baseline_metrics_dict must be a dictionary.")
    if not isinstance(stressed_metrics_dict, dict):
        raise TypeError("stressed_metrics_dict must be a dictionary.")

    all_metrics_keys = sorted(
        list(set(baseline_metrics_dict.keys()) | set(stressed_metrics_dict.keys())))

    baseline_values = []
    stressed_values = []

    for key in all_metrics_keys:
        baseline_values.append(baseline_metrics_dict.get(key, math.nan))
        stressed_values.append(stressed_metrics_dict.get(key, math.nan))

    data = {
        'Baseline': baseline_values,
        'Stressed': stressed_values
    }

    df = pd.DataFrame(data, index=pd.Index(all_metrics_keys, name='Metric'))

    return df


def generate_segment_impact_heatmap(segmentation_data_df, impact_data_series, segment_column):
    """
    Creates a heatmap visualization that illustrates the differential impact of a stress scenario across various
    predefined segments of the portfolio (e.g., rating band, sector, region). This highlights segments that
    are most vulnerable or resilient.

    Arguments:
        segmentation_data_df (pd.DataFrame): A DataFrame containing segmentation columns (e.g., 'sector', 'region').
        impact_data_series (pd.Series): A Series of individual prediction impacts or other relevant metrics.
        segment_column (str): The name of the column in segmentation_data_df to use for segmentation.

    Output:
        matplotlib.figure.Figure: A Matplotlib figure object displaying the heatmap.
    """

    # 1. Input Validation
    if not isinstance(segmentation_data_df, pd.DataFrame):
        raise TypeError("segmentation_data_df must be a pandas DataFrame.")
    if not isinstance(impact_data_series, pd.Series):
        raise TypeError("impact_data_series must be a pandas Series.")
    if not isinstance(segment_column, str):
        raise TypeError("segment_column must be a string.")

    if segment_column not in segmentation_data_df.columns:
        raise KeyError(
            f"Segment column '{segment_column}' not found in segmentation_data_df.")

    # 2. Data Alignment & Aggregation

    # Use a unique temporary column name for impact to avoid potential collisions.
    impact_col_name = '___segment_impact_data___'
    if impact_col_name in segmentation_data_df.columns:
        # Fallback for extremely rare collision
        impact_col_name = '___segment_impact_data_temp___'

    # Combine dataframes. Pandas will align impact_data_series with segmentation_data_df by their indices.
    # Rows in segmentation_data_df without corresponding index in impact_data_series
    # will result in NaN values for the new impact column.
    combined_df = segmentation_data_df.copy()
    combined_df[impact_col_name] = impact_data_series

    # Drop rows where the segment_column or the newly added impact_col_name has NaN values.
    # This ensures valid grouping and impact calculation.
    processed_df = combined_df.dropna(subset=[segment_column, impact_col_name])

    # Handle cases where processed_df becomes empty after cleaning (e.g., original DataFrame was empty,
    # or all impact values were NaN, or segment column was all NaN).
    if processed_df.empty:
        # Create a small figure for placeholder message
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.set_title(
            f"Segment Impact by '{segment_column}' (No Valid Data)", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No valid data to display heatmap.", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=12)
        # Call sns.heatmap with an empty DataFrame to ensure proper Matplotlib Figure structure
        sns.heatmap(pd.DataFrame(), ax=ax, cbar=False)
        plt.tight_layout()
        return fig

    # Group by the segment column and calculate the mean impact for each segment.
    segment_impact = processed_df.groupby(
        segment_column)[impact_col_name].mean()

    # Convert the aggregated Series to a DataFrame, which is suitable for seaborn.heatmap.
    # It will be a single-column DataFrame indexed by segment names.
    heatmap_data = segment_impact.to_frame(name='Average Impact')

    # 3. Heatmap Generation
    # Adjust figure size dynamically based on the number of segments for better readability.
    num_segments = len(heatmap_data)
    # Minimum height, plus 0.6 inches per segment
    fig_height = max(1.5, num_segments * 0.6)
    fig_width = 6  # Standard width for the heatmap

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        heatmap_data,
        ax=ax,
        annot=True,              # Show the numerical values on the heatmap cells
        fmt=".2f",               # Format annotations to two decimal places
        # Diverging colormap, suitable for showing positive/negative impacts
        cmap="coolwarm",
        linewidths=.5,           # Draw lines between cells
        linecolor='black',       # Color of lines between cells
        cbar_kws={'label': 'Average Impact'},  # Label for the color bar
        # Ensure y-axis labels (segment names) are shown
        yticklabels=True
    )

    ax.set_title(f"Average Impact by {segment_column}", fontsize=16)
    # Remove the x-axis label as it's implicitly 'Average Impact' from the column name
    ax.set_xlabel("")
    ax.set_ylabel(segment_column, fontsize=14)
    # Keep segment labels horizontal for better readability
    plt.yticks(rotation=0, fontsize=12)
    # Hide the x-tick label which would be 'Average Impact' for the single column
    plt.xticks([])

    plt.tight_layout()  # Adjust plot layout to prevent labels from overlapping
    return fig


def plot_delta_prediction_distribution(delta_predictions_series, threshold=None):
    """
    Generates a histogram or density plot to visualize the distribution of individual changes in predictions (Δŷ(s)).
    This helps in understanding the spread and skewness of the scenario's impact across the portfolio,
    and can highlight observations exceeding a defined threshold.
    """

    # Input validation: Ensure delta_predictions_series is a pandas Series or NumPy array.
    if not isinstance(delta_predictions_series, (pd.Series, np.ndarray)):
        raise TypeError(
            "delta_predictions_series must be a pandas.Series or numpy.ndarray.")

    # Create a new matplotlib figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle the case where delta_predictions_series is empty.
    # Display a message instead of an empty plot.
    if len(delta_predictions_series) == 0:
        ax.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='gray')
    else:
        # Plot a histogram to visualize the distribution of delta predictions.
        # 'bins='auto'' determines an optimal number of bins.
        # 'density=True' normalizes the histogram to form a probability density, resembling a density plot.
        ax.hist(delta_predictions_series, bins='auto', density=True, alpha=0.7, color='skyblue',
                label='Distribution of Δŷ(s)')

    # If a threshold is provided, highlight observations exceeding its magnitude.
    if threshold is not None:
        # Use the absolute value for the threshold magnitude.
        effective_threshold = abs(threshold)

        # Plot vertical dashed lines at the positive and negative threshold values.
        # The label is applied only to the positive threshold line to avoid duplicate legend entries for magnitude.
        ax.axvline(effective_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Magnitude Threshold: ±{effective_threshold}')
        ax.axvline(-effective_threshold, color='red',
                   linestyle='--', linewidth=2)

        # Shade the regions beyond the thresholds to visually emphasize values exceeding them.
        x_min, x_max = ax.get_xlim()
        # A small offset to ensure shading starts just after the vertical lines.
        epsilon = 1e-9
        ax.axvspan(effective_threshold + epsilon,
                   x_max, color='red', alpha=0.1)
        ax.axvspan(x_min, -effective_threshold - epsilon, color='red', alpha=0.1,
                   # Label for the shaded area in the legend.
                   label='Exceeds Threshold Region')

    # Set plot title and axis labels for better clarity.
    ax.set_title("Distribution of Individual Changes in Predictions (Δŷ(s))")
    ax.set_xlabel("Δŷ(s)")
    ax.set_ylabel("Density")
    # Add a subtle grid for readability.
    ax.grid(True, linestyle=':', alpha=0.6)

    # Display the legend if there are any labeled plot elements.
    # Check if any handles (plot elements with labels) exist.
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    # Adjust plot layout to prevent labels or titles from overlapping.
    plt.tight_layout()

    return fig
