# Optical Recognition of Handwritten Digits: SVM Optimization & EDA

This project applies Support Vector Machine (SVM) optimization to the Optical Recognition of Handwritten Digits dataset from the UCI Machine Learning Repository. The objective is to evaluate SVM performance across 10 different 70-30 train-test splits, optimize hyperparameters using RandomizedSearchCV (100 iterations per split), and report the best results. Exploratory Data Analysis (EDA) is performed to understand the dataset before modeling.

# Dataset Information

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)
- **Samples:** ~5,620
- **Features:** 64 (integer values, representing 8x8 pixel images)
- **Target:** Digit class (0–9)
- **Format:** Multiclass classification

# Exploratory Data Analysis (EDA)

Key EDA steps performed:

- **Data Inspection:** Checked shape, column names, and data types.
- **Missing Values:** Verified absence of missing values.
- **Correlation Analysis:** Explored correlations between features.
  ![image](https://github.com/user-attachments/assets/05bd6d4a-32e1-44d8-8686-fbf04aabaad9)

- **Sample Visualization:** Displayed a few digit images using matplotlib to understand the data visually.
  ![image](https://github.com/user-attachments/assets/48542aa1-b3a2-4a81-ba90-a04dd1e85f87)

- **Summary Statistics:** Used `describe()` to view mean, std, min, max for features.


# SVM Optimization Approach

- **Splitting:** The dataset was split into 10 different 70-30 train-test sets (using 10 random seeds).
- **Scaling:** StandardScaler was used for feature normalization.
- **Hyperparameter Optimization:** For each split, SVM hyperparameters were tuned using `RandomizedSearchCV` with 100 iterations:
  - Kernels: `linear`, `rbf`, `sigmoid`
  - C: log-uniform distribution [0.1, 100]
  - Gamma: `scale`, `auto`
- **Evaluation:** Best parameters and test accuracy were recorded for each split.
- **Convergence Curve:** For the split with the highest accuracy, the convergence curve (best accuracy vs. iteration) was plotted.

# Results

![image](https://github.com/user-attachments/assets/49bf24c3-50d1-42ce-819f-8ed8216b51c8)

![image](https://github.com/user-attachments/assets/5ea57715-6d4d-4c3a-b317-b0ffc78df9ee)


