# **`README.md`**

# Comparing Mixture, Box, and Wasserstein Ambiguity Sets in Distributionally Robust Asset Liability Management

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.08228-b31b1b.svg)](https://arxiv.org/abs/2602.08228)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2602.08228)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/distributionally_robust_asset_liability_management)
[![Discipline](https://img.shields.io/badge/Discipline-Financial%20Engineering%20%7C%20Operations%20Research-00529B)](https://github.com/chirindaopensource/distributionally_robust_asset_liability_management)
[![Data Sources](https://img.shields.io/badge/Data-Investing.com%20%7C%20CPP%20Reports-lightgrey)](https://www.investing.com/)
[![Core Method](https://img.shields.io/badge/Method-Distributionally%20Robust%20Optimization%20-orange)](https://github.com/chirindaopensource/distributionally_robust_asset_liability_management)
[![Analysis](https://img.shields.io/badge/Analysis-Stochastic%20Simulation%20%7C%20Sensitivity%20Analysis-red)](https://github.com/chirindaopensource/distributionally_robust_asset_liability_management)
[![Validation](https://img.shields.io/badge/Validation-Out--of--Sample%20Backtesting-green)](https://github.com/chirindaopensource/distributionally_robust_asset_liability_management)
[![Robustness](https://img.shields.io/badge/Robustness-Wasserstein%20Metric%20%7C%20Box%20Ambiguity-yellow)](https://github.com/chirindaopensource/distributionally_robust_asset_liability_management)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![CVXPY](https://img.shields.io/badge/CVXPY-%2300599C.svg?style=flat&logo=python&logoColor=white)](https://www.cvxpy.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/distributionally_robust_asset_liability_management)

**Repository:** `https://github.com/chirindaopensource/distributionally_robust_asset_liability_management`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Comparing Mixture, Box, and Wasserstein Ambiguity Sets in Distributionally Robust Asset Liability Management"** by:

*   **Alireza Ghahtarani** (HEC Montréal)
*   **Ahmed Saif** (Dalhousie University)
*   **Alireza Ghasemi** (Dalhousie University)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the ingestion and rigorous validation of financial market data to the formulation and solution of complex Distributionally Robust Optimization (DRO) models (Mixture, Box, Wasserstein), culminating in comprehensive out-of-sample evaluation and sensitivity analysis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_full_research_pipeline`](#key-callable-execute_full_research_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Ghahtarani et al. (2026). The core of this repository is the iPython Notebook `distributionally_robust_asset_liability_management_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline addresses the critical challenge of **Asset Liability Management (ALM)** for defined-benefit pension funds under conditions of severe parameter uncertainty and distributional ambiguity.

The paper proposes and compares three distinct DRO formulations against a traditional Stochastic Programming (SP) benchmark. This codebase operationalizes the proposed solution:
-   **Validates** data integrity using strict schema checks, temporal consistency enforcement, and economic plausibility bounds.
-   **Engineers** stochastic scenarios using Geometric Brownian Motion (GBM) calibrated to historical market regimes identified via $k$-means clustering.
-   **Solves** complex optimization problems: Linear Programs (LP) for SP, Mixture, and Box models; and Second-Order Cone Programs (SOCP) for the Wasserstein model.
-   **Evaluates** performance via rigorous out-of-sample backtesting, computing Funding Ratios, Fund Returns, and diversification metrics (HHI).

## Theoretical Background

The implemented methods combine techniques from Financial Engineering, Convex Optimization, and Robust Statistics.

**1. Asset Liability Management (ALM):**
The objective is to minimize the present value of contribution rates $y_t$ while ensuring the fund remains solvent ($A_t \ge \psi L_t$) over a horizon $T$.
$$ \min_{y, x} \mathbb{E}[W^\top y] \quad \text{s.t.} \quad \text{Solvency Constraints} $$

**2. Distributionally Robust Optimization (DRO):**
Instead of optimizing for a single expected value, DRO minimizes the worst-case expectation over an ambiguity set $\mathcal{P}$:
$$ \min_{x} \sup_{P \in \mathcal{P}} \mathbb{E}_P [f(x, \xi)] $$

**3. Ambiguity Sets Implemented:**
*   **Mixture Distribution (MD):** The ambiguity set is the convex hull of a finite set of likelihood distributions (regimes). Reformulated as a tractable LP using epigraph variables.
*   **Box Ambiguity (BD):** Probabilities are bounded within a hyper-rectangle around a nominal distribution. Reformulated as a tractable LP using Lagrangian duality.
*   **Wasserstein Metric (WM):** The ambiguity set contains all distributions within a Wasserstein distance $\epsilon$ of the empirical distribution. Reformulated as a tractable SOCP using dual norms (Mohajerin Esfahani & Kuhn, 2018).

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/distributionally_robust_asset_liability_management/blob/main/distributionally_robust_asset_liability_management_ipo_main.png" alt="DRO ALM System Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`distributionally_robust_asset_liability_management_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 32 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (actuarial assumptions, constraints, solver settings) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, temporal monotonicity, and numeric stability.
-   **Deterministic Execution:** Enforces reproducibility through strict seed control and immutable state management.
-   **Advanced Optimization:** Uses `CVXPY` to formulate and solve large-scale LPs and SOCPs, with support for commercial solvers (Gurobi, CPLEX) and open-source alternatives (CLARABEL, ECOS).
-   **Reproducible Artifacts:** Generates structured results containing raw time-series, aggregated metrics, and publication-ready figures.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Ingestion & Validation (Tasks 1-7):** Validates historical market data, cleanses time indices, handles missing values, and engineers fixed-income and risk-free return proxies.

2.  **Feature Engineering (Tasks 8-10):** Computes monthly log-returns, aggregates to annual simple returns, and constructs rolling feature vectors for regime identification.

3.  **Stochastic Modeling (Tasks 11-13):** Identifies market regimes (LV, MV, IHV, DHV) via $k$-means clustering, estimates regime-conditional GBM parameters ($\mu, \sigma$), and computes the cross-asset covariance matrix.

4.  **Scenario Generation (Tasks 14-16):** Generates in-sample training scenarios and frozen out-of-sample testing scenarios using Geometric Brownian Motion.

5.  **Ambiguity Set Materialization (Tasks 17-22):** Defines discount rate scenarios, computes PV wages/liabilities, constructs feasibility sets ($\mathcal{X}, \mathcal{Y}$), and materializes the specific inputs for MD, BD, and WM ambiguity sets (including Wasserstein radii and polyhedral supports).

6.  **Optimization (Tasks 23-26):** Formulates and solves the Stochastic Programming (SP), Mixture Distribution (MD), Box Distribution (BD), and Wasserstein Metric (WM) models.

7.  **Evaluation (Tasks 27-29):** Computes in-sample metrics (HHI), executes out-of-sample simulations to determine Funding Ratios and Fund Returns, and performs pairwise statistical significance testing.

8.  **Sensitivity Analysis (Task 31):** Executes a robustness sweep over the funding ratio threshold $\psi$ to analyze contribution rate sensitivity.

9.  **Final Output Generation (Task 32):** Aggregates all results into the final tables and figures matching the manuscript.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 32 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `execute_full_research_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_full_research_pipeline`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between validation, stochastic modeling, optimization, and evaluation modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `scikit-learn`, `cvxpy`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/distributionally_robust_asset_liability_management.git
    cd distributionally_robust_asset_liability_management
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy matplotlib pyyaml scikit-learn cvxpy
    ```

## Input Data Structure

The pipeline requires a single primary DataFrame:

1.  **`df_historical_raw`**:
    -   `Date` (datetime64[ns]): Monthly temporal index.
    -   `SP500`, `TSX_Comp`, `FTSE_300`, `Nikkei_225`, `SSE_Comp` (float64): Public Equity Indices.
    -   `PRIVEXD` (float64): Private Equity Index.
    -   `GSPRTRE` (float64): Real Estate Index.
    -   `TNX_Yield` (float64): 10-Year Treasury Yield (Fixed Income Proxy).
    -   `SPGTINTR` (float64): Infrastructure Index.
    -   `STOXX_Europe_20` (float64): European Private Equity Proxy.
    -   `RiskFree_Cash` (float64): Cash equivalent rate.

*Note: The pipeline includes a synthetic data generator for testing purposes if access to proprietary data is unavailable.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to use the top-level `execute_full_research_pipeline` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    # (Assumes config.yaml is in the working directory)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    df_historical_raw = generate_synthetic_market_data()

    # 3. Execute the entire replication study.
    master_report = execute_full_research_pipeline(df_historical_raw, config)
    
    # 4. Access results
    print(master_report.publication_outputs.table_5_oos_performance)
```

## Output Structure

The pipeline returns a `MasterExecutionReport` dataclass containing:
-   **`baseline_pipeline_artifacts`**: The complete state of the baseline ($\psi=1.05$) run, including all intermediate tensors and model solutions.
-   **`sensitivity_analysis_artifacts`**: The results of the robustness sweep over $\psi$, including the raw and formatted contribution rate tables.
-   **`publication_outputs`**: A dataclass containing the final publication-ready artifacts:
    -   `figure_1_asset_allocation`: Stacked bar charts of optimal weights.
    -   `figure_3_funding_ratio`: Line chart of OOS funding ratios.
    -   `figure_4_fund_return`: Line chart of OOS fund returns.
    -   `table_5_oos_performance`: Comprehensive OOS performance table.
    -   `table_3_pairwise_fr`: Pairwise t-test statistics for Funding Ratio.
    -   `table_4_pairwise_ret`: Pairwise t-test statistics for Fund Return.
    -   `table_2_sensitivity`: Sensitivity analysis of contribution rates.

## Project Structure

```
distributionally_robust_asset_liability_management/
│
├── distributionally_robust_asset_liability_management_draft.ipynb   # Main implementation notebook
├── config.yaml                                                      # Master configuration file
├── requirements.txt                                                 # Python package dependencies
│
├── LICENSE                                                          # MIT Project License File
└── README.md                                                        # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Actuarial Assumptions:** Initial assets ($A_0$), wages ($W_0$), and funding threshold ($\psi$).
-   **Regulatory Constraints:** Asset allocation limits and contribution rate bounds.
-   **Stochastic Modeling:** Number of scenarios ($K$), random seeds, and clustering hyperparameters.
-   **Ambiguity Sets:** Wasserstein radii heuristics, box perturbation bounds, and regime probabilities.
-   **Solver Settings:** Choice of solver (CLARABEL, ECOS, Gurobi), tolerances, and numerical scaling.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Risk Measures:** Incorporating CVaR or Entropic Value-at-Risk into the objective function.
-   **Dynamic Rebalancing:** Implementing multi-stage stochastic programming with recourse decisions.
-   **Alternative Ambiguity Sets:** Exploring Moment-based or Kullback-Leibler divergence ambiguity sets.
-   **Real-World Data Integration:** Connecting to live data feeds (Bloomberg, Refinitiv) for real-time ALM.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{ghahtarani2026comparing,
  title={Comparing Mixture, Box, and Wasserstein Ambiguity Sets in Distributionally Robust Asset Liability Management},
  author={Ghahtarani, Alireza and Saif, Ahmed and Ghasemi, Alireza},
  journal={arXiv preprint arXiv:2602.08228},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Comparing Mixture, Box, and Wasserstein Ambiguity Sets in Distributionally Robust Asset Liability Management: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/distributionally_robust_asset_liability_management
```

## Acknowledgments

-   Credit to **Alireza Ghahtarani, Ahmed Saif, and Alireza Ghasemi** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, CVXPY, and Scikit-Learn**.

--

*This README was generated based on the structure and content of the `distributionally_robust_asset_liability_management_draft.ipynb` notebook and follows best practices for research software documentation.*
