# Concrete Compressive Strength Analysis & Classification
**Course:** Machine Learning (ELL409/784)
**Assignment:** Programming Assignment 1
**Submission IDs:** 2025AIB2565, 2025AIB2558

---

## Project Overview
This project implements machine learning pipelines to analyze concrete compressive strength. It is divided into two stages:
1.  **Regression:** Predicting the exact numerical strength (MPa) based on mix composition and age.
2.  **Classification:** Determining if a concrete mix is "Fit for RCC" based on **IS 456:2000** standards (Strength ≥ 20 MPa AND Age ≥ 28 days).

The project includes **"From Scratch"** implementations of fundamental algorithms (Linear Regression via Normal Equation, Lasso via Coordinate Descent, Logistic Regression via Gradient Descent) alongside tuned library implementations (XGBoost, LightGBM, SVM).

---

## 📂 File Structure
* `2025AIB2565_2025AIB2558.ipynb`: The main Jupyter Notebook containing all code, visualizations, and model training.
* `ML_Report.pdf`: The comprehensive technical report analyzing the methods and results.
* `README.md`: This file providing execution instructions and project details.

---

## ⚙️ Prerequisites & Dependencies
The code is written in Python 3. To run the notebook, ensure the following libraries are installed.

**Required Libraries:**
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `imbalanced-learn` (for SMOTE)
* `ucimlrepo` (for automatic dataset downloading)

**Installation Command:**
You can install all dependencies via pip:
```bash
pip install -q ucimlrepo imbalanced-learn xgboost lightgbm numpy pandas matplotlib seaborn scikit-learn
```

## How to Run the Code

1.  **Install Dependencies**
    Open your terminal or command prompt and run the following command to install all required libraries:
    ```bash
    pip install ucimlrepo imbalanced-learn xgboost lightgbm numpy pandas matplotlib seaborn scikit-learn
    ```

2.  **Launch the Notebook**
    Open the file `2025AIB2565_2025AIB2558.ipynb` in **Jupyter Notebook**, **Jupyter Lab**, or **Google Colab**.

3.  **Data Download**
    Ensure you have an active internet connection when running the first cell. The code automatically fetches the dataset using:
    ```python
    from ucimlrepo import fetch_ucirepo
    concrete = fetch_ucirepo(id=165)
    ```

4.  **Execute the Pipeline**
    Run all cells in sequential order (Cell 1 to Final).
    * *Tip:* Use **Kernel > Restart & Run All** to ensure a clean execution environment.

5.  **Test the Deployment Function**
    The final cell of the notebook contains the `assess_concrete_mix()` function. You can manually input values (Cement, Water, Age, etc.) in that cell to see if a mix is "Fit for RCC" or not.

## Implementation Details

### Stage 1: Regression (Predicting Strength)
* **Linear Regression (From Scratch):** Implemented using the **Normal Equation** approach (function: `normal_equation`) to solve for weights analytically without loops.
* **Ridge Regression (From Scratch):** Implemented using the Normal Equation with an added L2 penalty matrix (function: `ridge_regression`).
* **Lasso Regression (From Scratch):** Implemented using **Coordinate Descent** with a soft-thresholding operator (function: `lasso_regression_from_scratch`).
* **Library Models:** SVR, XGBoost, and LightGBM were trained and tuned using `GridSearchCV`.

### Stage 2: Classification (RCC Suitability)
* **Target Creation:** The target variable `RCC_Suitable` was created based on the rule: `Strength >= 20 MPa` AND `Age >= 28 Days`.
* **Imbalance Handling:** **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training data to balance the "Fit" vs. "Not Fit" classes.
* **Logistic Regression (From Scratch):** Implemented via a custom class `LogisticRegressionScratch` using **Gradient Descent**. It supports both L1 and L2 regularization penalties.
* **Library Models:** Random Forest, Gradient Boosting, SVM, and Decision Tree were evaluated, with Random Forest achieving the best F1 score.