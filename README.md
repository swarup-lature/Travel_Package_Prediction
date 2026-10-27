# Travel_Package_Prediction

An end-to-end Machine Learning classification pipeline built to predict whether a customer will purchase a travel package based on demographic, behavioral, and engagement metrics.

## 🚀 Project Overview
Identifying prospective buyers for travel packages helps optimize marketing campaigns and resource allocation. This project cleans raw customer data, engineers meaningful aggregate features, handles missing values, and evaluates multiple classification algorithms to determine the best predictive model.

---

## 📊 Dataset & Preprocessing
* **Source File:** `Travel.csv`
* **Data Cleaning:**
  * Fixed categorical typos (e.g., standardizing `"Fe Male"` to `"Female"`).
  * Merged marital status categories (`"Single"` mapped to `"Unmarried"`).
  * Handled missing values across features using appropriate imputation strategies (Mean/Median for continuous variables, Mode for categorical features).
  * Removed unique identifier columns (`CustomerID`) that carry no predictive power.
* **Feature Engineering:**
  * Created a combined feature **`TotalVisiting`** by summing `NumberOfPersonVisiting` and `NumberOfChildrenVisiting`.

---

## 🤖 Model Performance & Evaluation
Several classification models were trained and evaluated on an 80-20 train-test split:

| Model | Testing Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 93.35% | 97.01% | 68.06% | 0.8000 |
| **Decision Tree** | 91.10% | 77.96% | 75.92% | 0.7692 |
| **AdaBoost** | 84.05% | 78.69% | 25.13% | 0.3810 |
| **Gradient Boosting** | 85.89% | 77.32% | 39.27% | 0.5208 |
| **XGBoost** | **94.68%** | **95.42%** | **76.44%** | **0.8488** |

* **Best Performer:** **XGBoost** achieved the highest overall testing accuracy (**94.68%**) and the strongest balance between precision and recall (F1-score of **0.85**).
* **Hyperparameter Tuning:** Optimized via `RandomizedSearchCV` (`max_depth=10`, `n_estimators=200`, `learning_rate=0.2`), securing a cross-validation score of **93.3%**.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** 
  * Data Manipulation & Analysis: `Pandas`, `NumPy`
  * Machine Learning: `Scikit-Learn`, `XGBoost`
  * Visualization: `Matplotlib`, `Seaborn`

---

## ⚙️ Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/swarup-lature/Traval_Package.git](https://github.com/swarup-lature/Traval_Package.git)
