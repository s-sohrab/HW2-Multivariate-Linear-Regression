

```markdown
# Multivariate Linear Regression — House Price Prediction

A clean, end-to-end implementation of **multivariate linear regression** using `scikit-learn`, built as the second assignment (HW2) of a Machine Learning course. The notebook predicts house prices from three features — **area**, **number of rooms**, and **building age** — and walks through data generation, exploratory analysis, model training, evaluation, coefficient interpretation, and a multicollinearity check.

---

##  Problem Statement

The goal is to model the relationship between multiple input features and a continuous target, and to understand how each feature contributes to the prediction. The assignment specifically highlights three learning objectives:

- **Modeling** the relationship between several features and a single output.
- **Practical application**: analyzing more complex real-estate scenarios.
- **Challenge**: handling correlation between features (*multicollinearity*).
- **Importance**: understanding the effect of multiple features on a linear model.

---

##  Objective

Train a linear regression model that predicts **house price** from:

| Feature | Description | Range |
|---------|-------------|-------|
| Area | House area in square meters | 50 – 200 |
| Rooms | Number of rooms | 1 – 4 |
| Age | Building age (years) | 1 – 19 |

The ground-truth data-generating process is:

```
price = 50 × area + 2000 × rooms − 100 × age + ε
```

where `ε ~ N(0, 500²)` is additive Gaussian noise. A well-fit model should recover coefficients close to **(50, 2000, −100)** and an intercept close to **0**.

---

##  Repository Structure

```
HW2-Multivariate-Linear-Regression/
├── ML_HW2_Multivariate_LinearRegression.ipynb   # Main notebook (implementation + results)
└── README.md                                     # Project documentation
```

---

##  Methodology

The notebook follows a clean, reproducible pipeline:

1. **Environment setup** — imports and reproducibility seed (`random_state=42`).
2. **Synthetic data generation** — 100 samples built with `np.column_stack`, `np.linspace`, and `np.random.randint`, plus Gaussian noise.
3. **Exploratory Data Analysis (EDA)** — descriptive statistics, feature–target correlation, and scatter plots.
4. **Train/test split** — 80% training, 20% testing (`train_test_split`).
5. **Model training** — `LinearRegression` from `scikit-learn`.
6. **Evaluation** — Mean Squared Error (MSE) and R² score on both train and test sets.
7. **Coefficient analysis** — comparison of learned coefficients against the ground-truth values.
8. **Multicollinearity diagnostics** — correlation matrix heatmap and Variance Inflation Factor (VIF) via `statsmodels`.

---

##  Results

The model is evaluated on both the training and test sets to detect underfitting/overfitting. Key outputs include:

- **Learned coefficients** compared to the true coefficients `(50, 2000, −100)`.
- **Learned intercept** compared to the true intercept `0`.
- **MSE** and **R²** on train and test splits.
- **Correlation matrix** and **VIF** values for each feature.

> Because the data is generated from a genuinely linear process with independent features, the model is expected to achieve a **high R²**, **low MSE**, **coefficients very close to the ground truth**, and **negligible multicollinearity** (correlations near 0, VIF near 1).

---

##  Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Programming language |
| NumPy | Array operations and synthetic data generation |
| pandas | Tabular data handling |
| Matplotlib / Seaborn | Visualization (scatter plots, correlation heatmap) |
| scikit-learn | Linear regression, train/test split, metrics |
| statsmodels | Variance Inflation Factor (VIF) |
| Google Colab | Development and execution environment |
| GitHub | Version control and sharing |

---

##  How to Run

### Option 1 — Open directly in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s-sohrab/HW2-Multivariate-Linear-Regression/blob/main/ML_HW2_Multivariate_LinearRegression.ipynb)

### Option 2 — Run locally

```bash
git clone https://github.com/s-sohrab/HW2-Multivariate-Linear-Regression.git
cd HW2-Multivariate-Linear-Regression
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
jupyter notebook ML_HW2_Multivariate_LinearRegression.ipynb
```

Then run all cells from top to bottom (`Runtime → Run all` in Colab, or `Cell → Run All` in Jupyter).

---

##  Key Takeaways

- Linear regression recovers the true underlying coefficients almost exactly when the data-generating process is linear and features are independent.
- R² and MSE together give a clearer picture than either metric alone — R² for explained variance, MSE for absolute error magnitude.
- Multicollinearity diagnostics (correlation matrix + VIF) are essential when interpreting coefficients; here, low VIF values confirm that each feature contributes independently.
- Real-world datasets rarely satisfy these ideal conditions — this exercise builds the baseline intuition needed to recognize and handle violations.

---

## 🙏 Acknowledgements

This assignment was completed as part of a Machine Learning course. Sincere thanks to:

- **Dr. Mohammad Ali Javadzadeh** — course instructor, for the assignment design and guidance on multivariate regression and multicollinearity.
- **Dr. Ali Jaleh Karimi** — teaching assistant, for continuous support throughout the learning process.

Their instruction and feedback made this exercise a valuable learning experience.

---

##  License

This repository is shared for **educational purposes** as part of a course assignment. Please credit the author if you reuse or adapt the material.

---

## 👤 Author

**Samad Sohrab** — [@s-sohrab](https://github.com/s-sohrab)
