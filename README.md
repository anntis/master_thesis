### 🌍Migration Policy and Environmental Impact: Comparing Double Machine Learning and Bayesian Inference

This repository contains the code and data used for a Master's thesis analyzing the **impact of national migration policy on municipal waste generation** across European Union countries.
This study utilizing both Double Machine Learning (DML) and Double Robust Bayesian (DR Bayesian) methods for robust estimation of the Average Treatment Effect (ATE).

To replicate the full analysis and obtain all model estimates, follow the workflow outlined below.

| Step | File Name | Description | Output |
| :--- | :--- | :--- | :--- |
| **1. Data Preparation** | `main_code.ipynb` | Processes the raw data and calculates the **propensity scores** required for causal modeling. | Full dataset (including PS) saved as a `.mat` file. |
| **2. Bayesian Estimation** | `main.m` (MATLAB) | Performs the **Bayesian ATE estimation** using the `.mat` file generated in Step 1. | Bayesian ATE estimates. |
| **3. Comparison & Visualization** | `methods_plot.ipynb` | Generates and plots the **estimates distributions** for the pooled DML models (Lasso, RF, XGBoost) and all Bayesian models (Bayes, PA Bayes, DR Bayes). | Comparative visualization (PDFs). |
| **4. Panel Data Inference** | `panel_data_workflow.ipynb` | Implements specialized DML methods for panel data (CRE DML and DML with dummies) and produces their corresponding distribution graphs. | Panel data ATE estimates and distributions. |

The analysis confirms that a country’s attitude toward migrants has an impact on generated amount of municipal waste. It shows that municipal waste production in countries that are categorized as "welcoming to migrants" is higher than in those with stricter migration policies.