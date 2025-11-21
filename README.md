This comprehensive `README.md` provides an overview of the "QuLab - Scenario-Based Model Robustness Test" Streamlit application, designed for financial ML model stress testing.

---

# QuLab - Scenario-Based Model Robustness Test

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab - Scenario-Based Model Robustness Test** is a Streamlit application developed as a lab project to explore scenario-based robustness testing of financial Machine Learning (ML) models.

The core idea behind this application is to apply structured stress transformations, denoted as $T_s(x)$, directly to the input features of a trained ML model. This allows us to evaluate how a model's predictions, $f_\theta(x^{(s)})$, change when exposed to transparent and parameterized shocks (e.g., macro-economic, market-specific, or borrower-level stresses) without the need for retraining the model for each new scenario.

This methodology is particularly relevant for:
*   **Quantitative analysts and model developers** involved in credit, market, or trading model development.
*   **Risk managers and validators** performing robustness assessments aligned with regulatory standards like SR 11-7.
*   **CFA/FRM/PRM candidates** seeking to bridge theoretical stress testing concepts with practical ML implementations.

The application treats a pre-trained model as an entity subject to various "what-if" scenarios, where each scenario is formally defined as a transformation $T_s$ applied to the model's feature vector. The general form of the transformation is $x^{(s)} = T_s(x) = x \odot (1 + \delta_s)$, where $\delta_s$ is a vector of scenario-specific shocks.

## Features

This application is structured into three main pages, accessible via the sidebar navigation:

1.  **Stress Transformation Library ($T_s(x)$)**:
    *   **Conceptual Overview**: Introduces the mathematical formulation of stress transformations ($x^{(s)} = x \odot (1 + \delta_s)$) and explains its components.
    *   **Scenario Types**: Details typical financial scenario types that can be modeled using these transformations, such as macro downturns, market stresses, and idiosyncratic borrower shocks.
    *   **Theoretical Foundation**: Provides the theoretical basis for feature-level transformations in ML model stress testing.

2.  **Scenario Definition Panel & Model Response Measurement**:
    *   **Interactive Scenario Definition**: Allows users to define custom stress scenarios using sliders to adjust scale factors (multipliers) for key financial features (e.g., income, debt-to-income ratio, utilization rate).
    *   **Dummy Model Simulation**: Utilizes a synthetic dataset and a simple Logistic Regression model (e.g., for Probability of Default) to demonstrate the impact of stress.
    *   **Prediction Comparison**: Calculates and displays both baseline predictions ($f_\theta(x)$) and stressed predictions ($f_\theta(x^{(s)})$) for the model.
    *   **Impact Metrics**: Computes the mean change in predictions ($\Delta \hat{y}^{(s)}$), such as the mean change in Probability of Default (PD).
    *   **Distribution Analysis**: Visualizes the distribution of individual changes in predictions ($\Delta \hat{y}^{(s)}$) using an interactive histogram, showcasing the spread of impacts across the portfolio.

3.  **Portfolio Aggregation & Sensitivity Analysis**:
    *   **Portfolio-Level Metrics**: Aggregates individual model responses to derive crucial portfolio-level metrics, such as the overall mean Probability of Default (PD) and changes in it under stress.
    *   **Risk Grade Migration**: Presents a "Risk Grade Migration" matrix, illustrating how the percentage of obligors shift between different risk categories (e.g., Low, Medium, High Risk) from baseline to stressed conditions.
    *   **Sensitivity Analysis**: Conducts a gradual stress path analysis by varying the intensity of the scenario shock ($\alpha$) from 0 (no stress) to 1 (full stress). The transformation is defined as $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$.
    *   **Nonlinearity Visualization**: Plots the trajectory of mean stressed PD as $\alpha$ increases, allowing identification of nonlinear model behaviors, sensitivity thresholds, and potential instabilities.

## Getting Started

Follow these instructions to get the QuLab application up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository** (if this were a git repo):
    ```bash
    git clone https://github.com/yourusername/qu-lab-stress-testing.git
    cd qu-lab-stress-testing
    ```
    *(Note: Replace `https://github.com/yourusername/qu-lab-stress-testing.git` with the actual repository URL if available.)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit>=1.0.0
    pandas>=1.0.0
    numpy>=1.20.0
    scikit-learn>=1.0.0
    plotly>=5.0.0
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

After installation, you can launch the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your default web browser. If it doesn't open automatically, Streamlit will provide a local URL (usually `http://localhost:8501`) that you can navigate to.

## Usage

The application provides a user-friendly interface through Streamlit.

1.  **Navigation**: Use the sidebar on the left to navigate between the three main sections of the lab:
    *   **Stress Transformation Library**: Understand the theoretical concepts.
    *   **Scenario Definition & Model Response**: Define your stress scenario and observe individual model impacts.
    *   **Portfolio Aggregation & Sensitivity**: Analyze aggregated portfolio effects and model sensitivity to varying stress intensities.

2.  **Scenario Definition**: On the "Scenario Definition & Model Response" page, use the sliders to set the desired stress multipliers for different financial features. The dummy model will automatically re-calculate predictions based on your inputs.

3.  **Sensitivity Analysis**: On the "Portfolio Aggregation & Sensitivity" page, you can define a specific base shock and then adjust the "Stress Intensity ($\alpha$)" slider to observe how the mean portfolio PD changes along a gradual stress path.

## Project Structure

The project is organized as follows:

```
.
├── app.py                      # Main Streamlit application entry point
├── application_pages/          # Directory containing individual page modules
│   ├── __init__.py             # Initializes the application_pages package
│   ├── page1.py                # "Stress Transformation Library" page logic
│   ├── page2.py                # "Scenario Definition & Model Response" page logic
│   └── page3.py                # "Portfolio Aggregation & Sensitivity" page logic
├── requirements.txt            # List of Python dependencies
└── README.md                   # This README file
```

## Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (for interactive web application development)
*   **Backend / Logic**: [Python 3.x](https://www.python.org/)
*   **Data Manipulation**: [pandas](https://pandas.pydata.org/)
*   **Numerical Operations**: [NumPy](https://numpy.org/)
*   **Machine Learning**: [scikit-learn](https://scikit-learn.org/stable/) (for the dummy Logistic Regression model and data scaling)
*   **Plotting**: [Plotly](https://plotly.com/python/) (for interactive visualizations)

## Contributing

As this is a lab project, direct contributions might not be formally managed as a typical open-source project. However, if you have suggestions, find issues, or wish to enhance the code:

1.  Fork the repository (if applicable).
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes and ensure they are well-tested.
4.  Submit a pull request with a clear description of your modifications.

## License

This project is intended for educational and lab purposes by Quant University. The code is provided "as is" without warranty of any kind. For specific licensing terms or usage outside of Quant University lab activities, please contact the institution.

## Contact

For questions or inquiries related to this QuLab project, please refer to the Quant University resources or contact their support channels.

Quant University Website: [https://www.quantuniversity.com/](https://www.quantuniversity.com/)

---