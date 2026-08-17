# Personalised Machine Learning for Software Task Effort Estimation

This project investigates whether **personalised machine-learning models can estimate software task effort more accurately than developers' own estimates**, with the broader aim of understanding whether improved estimation can help reduce the planning fallacy in individual task scheduling.

The project uses the publicly available **SiP (Software effort estimation) dataset** and implements a leakage-aware, time-ordered machine-learning workflow covering data preparation, exploratory analysis, baseline modelling, advanced modelling, statistical comparison and scheduling simulation.

## Research Question

> **To what extent can personalised machine-learning models estimate software task effort more accurately than developers' own estimates and thereby reduce the planning fallacy in individual task scheduling?**

---

## Dataset

The project uses the **SiP effort estimation dataset** by Derek M. Jones.

- Dataset repository: https://github.com/Derek-Jones/SiP_dataset
- `Sip-task-info.csv` — task, project, developer and effort information
- `est-act-dates.csv` — estimation, start and completion dates

The notebook downloads the two source files directly from the GitHub repository:

```text
https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/Sip-task-info.csv
https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/est-act-dates.csv
```

The original dataset repository describes the data as the **SiP effort estimation dataset** and points to published work describing its analysis.

---

## Project Aim

The project aims to determine whether machine-learning models that incorporate developer-specific historical information can improve software effort estimation compared with developers' original estimates.

The analysis specifically evaluates:

1. Developers' original effort estimates as the human baseline.
2. Non-personalised machine-learning models.
3. Personalised machine-learning models.
4. Direct effort prediction.
5. Residual/error correction of the original human estimate.
6. Time-aware historical baselines.
7. Whether improvements are consistent across developers and effort levels.
8. The implications of improved prediction for task scheduling and overbooking.

---

## Main Methodology

The notebook is organised into four analytical stages.

### Week 1 — Data Understanding and Cohort Construction

- Downloads and verifies the SiP dataset.
- Audits duplicates and repeated task numbers.
- Reconstructs task-level records.
- Parses and validates dates.
- Checks data quality and chronological validity.
- Defines the primary single-developer cohort.
- Establishes developers' original estimates as the human baseline.
- Assesses whether the dataset contains sufficient repeated developer histories for personalised modelling.
- Produces initial exploratory visualisations and evidence reports.

### Week 2 — Feature Engineering and Baseline Modelling

- Performs deeper exploratory analysis.
- Investigates extreme effort values without automatically removing them.
- Creates features available at estimation time.
- Builds leakage-safe historical features.
- Uses strictly earlier estimation dates when calculating historical information.
- Creates a chronological train/validation/test split.
- Performs a target-leakage audit.
- Builds naïve historical baselines.
- Tests Linear Regression and Ridge Regression.
- Compares direct effort prediction with residual/error correction.
- Performs effort-band and outlier sensitivity analysis.

### Week 3 — Advanced Machine Learning

Three advanced model families are evaluated:

- **Random Forest**
- **XGBoost**
- **Neural Network (MLP)**

Each model is evaluated under:

- **Non-personalised scope**
- **Personalised scope**

Two prediction strategies are used:

1. **Direct log-effort prediction**
2. **Residual correction**

For residual correction, the model predicts:

```text
HoursActual - HoursEstimate
```

The predicted correction is then added to the developer's original estimate.

Validation data is used for model/configuration selection and calibration. The final test period remains fixed and is not used for model selection.

### Week 4 — Scheduling Simulation and Practical Evaluation

The final stage uses the verified Week 3 evidence to examine practical scheduling implications.

The analysis includes:

- cumulative scheduling simulations;
- weekly scheduling scenarios;
- high-volume developer cohorts;
- fixed safety-buffer sensitivity scenarios;
- overbooking;
- unused capacity;
- practical comparison of improved prediction against scheduling risk.

The final conclusion distinguishes between **better prediction accuracy** and a **guaranteed reduction in overbooking**.

---

## Data Leakage Controls

A major design principle of this project is preventing future information from influencing predictions.

Historical features are calculated using information from **strictly earlier estimation dates**.

The following outcome-derived variables are not used as predictors:

- `HoursActual`
- `DeveloperHoursActual`
- `TaskPerformance`
- `DeveloperPerformance`
- `EstimateErrorHours`
- `AbsoluteEstimateErrorHours`
- `EstimateDirection`
- `UnderIndicator`
- `StartedOn`
- `CompletedOn`
- `StatusCode`
- developer-hour aggregates

The main evaluation uses a **chronological split rather than a random split**, so later tasks are evaluated as future observations.

The fixed Week 3 cohort contains:

- **7,042 tasks**
- **20 developers**
- **18 projects**
- **4,930 training tasks**
- **1,057 validation tasks**
- **1,055 test tasks**

---

## Model Evaluation

The main regression metrics are:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual effort.

Lower MAE is better.

### Root Mean Squared Error (RMSE)

Penalises larger prediction errors more heavily.

Lower RMSE is better.

### Median Absolute Error

Provides a more robust description of typical prediction error.

### Additional Analysis

The project also evaluates:

- underestimation rate;
- overestimation rate;
- exact-estimate rate;
- developer-level improvement;
- actual-effort bands;
- feature importance;
- paired bootstrap confidence intervals;
- Wilcoxon signed-rank testing;
- scheduling overbooking;
- unused capacity.

---

## Key Verified Findings

The completed notebook reports the following final findings:

- The strongest personalised model was **personalised XGBoost**.
- Personalised XGBoost improved test **MAE by 8.61%** relative to developers' estimates.
- Personalised XGBoost improved test **RMSE by 17.29%** relative to developers' estimates.
- The high-volume scheduling cohort contained **1,044 tasks from three developers**.
- Aggregate schedule-horizon MAE improved by **66.96%** in that scheduling analysis.
- However, only **one of the three** high-volume developers improved in that particular cohort.
- Unbuffered machine-learning scheduling increased weekly overbooking.
- Fixed contingency buffers reduced overbooking, but also created additional unused capacity.

Therefore, the defensible conclusion is that:

> **Personalised machine learning can mitigate, but not eliminate, software task planning and estimation error.**

Improved predictive accuracy should not be interpreted as a guarantee that organisational scheduling problems will disappear.

---

## Project Structure

A recommended repository structure is:

```text
.
├── README.md
├── SiP_complete (1).ipynb
│
├── data/
│   ├── raw/
│   │   ├── Sip-task-info.csv
│   │   └── est-act-dates.csv
│   │
│   └── processed/
│       └── sip_week2_feature_table.csv
│
├── models/
│   └── trained model artefacts
│
├── outputs/
│   ├── baseline metrics
│   ├── model results
│   ├── predictions
│   ├── feature importance
│   ├── statistical comparisons
│   └── summary JSON/CSV files
│
├── figures/
│   ├── exploratory plots
│   ├── MAE/RMSE comparisons
│   ├── developer comparisons
│   └── feature-importance plots
│
└── reports/
    └── generated analysis reports
```

The exact folders generated by the notebook may differ slightly depending on whether it is executed in Google Colab or locally.

---

## Technologies and Libraries

The project is implemented in Python and uses:

- Python 3
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- XGBoost
- Joblib
- Requests
- Jupyter Notebook / Google Colab

Main machine-learning components include:

```text
Linear Regression
Ridge Regression
Random Forest Regressor
XGBoost Regressor
MLPRegressor / Neural Network
```

---

## Installation

Install the main dependencies with:

```bash
pip install pandas numpy matplotlib scipy scikit-learn xgboost joblib requests jupyter
```

For Google Colab, the notebook can be opened directly and the required packages can be installed in a setup cell if they are not already available.

---

## Running the Project

### Option 1 — Google Colab

1. Open Google Colab.
2. Upload `SiP_complete (1).ipynb`.
3. Run the notebook **from the first cell to the final cell in order**.
4. Allow the notebook to download the SiP source CSV files.
5. Review the generated outputs, figures and reports.

The notebook detects Google Colab and uses project directories under `/content/`.

### Option 2 — Local Jupyter

1. Clone or download the project.
2. Install the dependencies.
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Run all cells sequentially.

Example:

```bash
jupyter notebook
```

Then open:

```text
SiP_complete (1).ipynb
```

---

## Important Reproducibility Notes

The notebook is designed as a sequential research artefact.

Please **do not skip directly to Week 3 or Week 4** unless the required outputs from the preceding stage already exist.

The workflow deliberately carries evidence forward:

```text
Week 1
   ↓
Cohort + baseline + data-quality evidence
   ↓
Week 2
   ↓
Leakage-safe features + fixed chronological split
   ↓
Week 3
   ↓
Advanced models + final test evaluation
   ↓
Week 4
   ↓
Scheduling simulation + practical evaluation
```

The random seed is set to:

```text
42
```

The final test period is fixed before advanced model evaluation.

No configuration or calibration setting is selected after inspecting test performance.

---

## Human Baseline vs Machine Learning

The human baseline is defined as:

```python
Prediction = HoursEstimate
```

The actual observed outcome is:

```python
Target = HoursActual
```

This means the machine-learning models are not evaluated in isolation. Their performance is directly compared against the estimate originally produced by the developer.

This is important because the research question is about whether personalised machine learning can provide a meaningful improvement over the existing human estimation process.

---

## Personalisation

The personalised models include developer-specific information such as:

- `DeveloperID`
- prior developer task count;
- prior developer actual effort;
- prior developer estimation error;
- prior developer MAE;
- prior developer underestimation rate;
- developer-project historical information.

These historical variables are calculated using only information available before the task being predicted.

The non-personalised models exclude developer identity and developer-specific historical variables, allowing the project to assess whether personalisation itself contributes to predictive performance.

---

## Statistical Comparison

The strongest advanced model is compared with developers' estimates using task-level paired errors.

Two complementary statistical procedures are implemented:

1. **Paired bootstrap confidence interval** for the difference in MAE.
2. **One-sided Wilcoxon signed-rank test** assessing whether human absolute errors are larger than model absolute errors.

Statistical testing is treated as complementary evidence rather than a replacement for practical error comparisons.

---

## Feature Importance

For the strongest tree-based model, feature importance is extracted and visualised.

Feature importance is used for interpretation only.

> Feature importance indicates which variables the fitted model used strongly; it does not establish causality.

---

## Scheduling Simulation

The final stage explores whether improved effort predictions translate into improved scheduling decisions.

The simulation considers:

- developer task histories;
- rolling scheduling scenarios;
- predicted task effort;
- actual task effort;
- schedule horizon error;
- weekly overbooking;
- unused capacity;
- fixed safety buffers.

The scheduling results are interpreted cautiously because better prediction does not automatically imply better operational outcomes.

---

## Limitations

Several limitations should be considered when interpreting the results:

- The analysis uses a publicly available historical software-effort dataset rather than newly collected organisational data.
- Developer identities are anonymised.
- The primary personalised cohort focuses on tasks involving one developer.
- Performance may vary across developers.
- The scheduling simulation is descriptive and scenario-based.
- Fixed safety buffers are sensitivity scenarios rather than test-optimised policies.
- Model performance does not establish causal relationships between personalisation and planning outcomes.
- The results should not be assumed to generalise to every software organisation or development environment.

---

## Reproducibility and Research Integrity

The project follows several safeguards:

- chronological train/validation/test evaluation;
- leakage-safe historical feature construction;
- fixed test-period evaluation;
- predeclared model configuration sets;
- validation-based model selection;
- separation of primary and sensitivity analyses;
- explicit documentation of excluded outcome variables;
- preservation of the full valid cohort for primary modelling;
- qualified interpretation of scheduling results.

---

## Source and Acknowledgement

The underlying SiP dataset is provided by **Derek M. Jones** through the public SiP dataset repository.

Dataset repository:

https://github.com/Derek-Jones/SiP_dataset

The repository states that the data are described in the paper:

**Derek M. Jones and Stephen Cullum, "A conversation around the analysis of the SiP effort estimation dataset."**

Users of the dataset should consult the original repository and associated publication for the appropriate dataset context and citation requirements.

---

## Citation

If you use this project or the underlying dataset in academic work, cite the original SiP dataset source and the associated publication. Please also acknowledge the original dataset authors.

Original dataset:

```text
Jones, D. M. — SiP effort estimation dataset
https://github.com/Derek-Jones/SiP_dataset
```

---

## Author

**Software Task Effort Estimation — Personalised Machine Learning Project**

This repository contains the computational artefact supporting the research investigation into personalised machine-learning-based software effort estimation and its implications for individual task scheduling.
