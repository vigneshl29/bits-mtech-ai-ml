# Student Dropout Classification

## a. Problem Statement

The objective of this project is to build and compare machine learning classification models for predicting a student's final academic outcome. The target variable contains three classes:

- **Dropout**
- **Enrolled**
- **Graduate**

Five classification algorithms are trained using the same train/test split and preprocessing pipeline. Their performance is evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The notebook uses an **80% training / 20% testing split** with `random_state=42`. The dataset contains 4,424 student records and 36 input features.

---

## b. Dataset Description

The project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository (Dataset ID: 697).

The dataset contains information about students enrolled in different undergraduate programmes. The features cover:

- Demographic information such as marital status, gender, age, and nationality.
- Admission and academic-path information such as application mode, application order, course, and previous qualification.
- Socio-economic information such as parents' qualifications and occupations, scholarship status, tuition-fee status, debtor status, and displacement.
- Academic performance information from the first and second semesters, including enrolled, evaluated, approved, and credited curricular units and semester grades.
- Macroeconomic indicators including unemployment rate, inflation rate, and GDP.

### Dataset Statistics

| Property | Value |
|---|---:|
| Number of records | 4,424 |
| Number of original features | 36 |
| Target classes | 3 |
| Missing values | 0 |
| Training records | 3,539 |
| Testing records | 885 |

### Target Distribution

| Target Class | Count | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |

The notebook also performs feature engineering, increasing the feature count from **36 to 46**. The engineered features include semester approval/evaluation rates, total approved and enrolled units, average semester grade, grade change, overall approval rate, and a financial/administrative risk indicator.

The dataset is a multiclass classification problem with noticeable class imbalance, so macro-averaged evaluation metrics are used for model comparison.

Dataset source: UCI Machine Learning Repository, *Predict Students' Dropout and Academic Success*, Dataset ID 697.

---

## c. GitHub Repository Link

**GitHub Repository:** `[https://github.com/vigneshl29/bits-mtech-ai-ml/tree/main/ML_Assignment_2]`

---

## d. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

The same train/test split and preprocessing strategy were used for all five models.

### Model Configuration

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=8`, `min_samples_leaf=5`, `class_weight="balanced"` |
| KNN | `n_neighbors=15`, `weights="distance"` |
| Naive Bayes | Gaussian Naive Bayes |
| Support Vector Machine | RBF kernel, `C=2.0`, `class_weight="balanced"`, `probability=True` |

### Evaluation Metrics Comparison

The notebook calculates Precision, Recall, and F1 using **macro averaging**, and calculates multiclass AUC using **one-vs-rest (OvR), macro averaging**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7254 | 0.8810 | 0.6918 | 0.6969 | 0.6865 | 0.5727 |
| Decision Tree | 0.6859 | 0.8387 | 0.6765 | 0.6834 | 0.6608 | 0.5324 |
| KNN | 0.7571 | 0.8553 | 0.7229 | 0.6556 | 0.6706 | 0.5985 |
| Naive Bayes | 0.2452 | 0.7110 | 0.4688 | 0.3938 | 0.2187 | 0.1270 |
| Support Vector Machine | 0.7322 | 0.8820 | 0.7007 | 0.7043 | **0.6938** | 0.5816 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | A strong linear baseline. The standardized and one-hot encoded features provide good separation of the three outcomes using a linear decision boundary. It achieved **72.54% accuracy** and a **0.6865 macro F1**. |
| **Decision Tree** | Captures nonlinear interactions and threshold effects. The depth and minimum-leaf constraints help reduce overfitting. It achieved **68.59% accuracy** and a **0.6608 macro F1**, making it weaker than the other main models. |
| **KNN** | Uses local similarity in the transformed feature space, making feature scaling important. KNN achieved the **highest accuracy (75.71%)** and **highest MCC (0.5985)**, but its macro F1 (**0.6706**) was lower than Logistic Regression and SVM because its recall for the Enrolled class was relatively weak. |
| **Naive Bayes** | Provides a fast probabilistic baseline based on conditional independence assumptions. Its performance was substantially lower, with **24.52% accuracy** and **0.2187 macro F1**, indicating that the assumptions of the model were not well suited to this feature set. |
| **Support Vector Machine** | Uses an RBF kernel to model nonlinear class boundaries. Scaling and class weighting help it handle heterogeneous features and class imbalance. It achieved the **highest macro F1 (0.6938)**, along with **73.22% accuracy**, **0.8820 AUC**, and **0.5816 MCC**. |

---

## Overall Winner for the Dataset

### **Support Vector Machine (SVM)**

The notebook selects the winner using the **highest macro F1 score**. Based on this criterion:

- **Winner:** Support Vector Machine
- **Macro F1:** **0.6938**
- **Accuracy:** **0.7322**
- **AUC:** **0.8820**
- **Macro Precision:** **0.7007**
- **Macro Recall:** **0.7043**
- **MCC:** **0.5816**

Although **KNN achieved the highest accuracy and MCC**, SVM achieved the highest macro F1 score. Since the dataset is imbalanced across the three classes, macro F1 provides a more balanced view of performance across Dropout, Enrolled, and Graduate classes. Therefore, **SVM is the overall winner according to the notebook's selection criterion**.

---

## Project Files

The notebook generates the following trained model files:

```text
model/
├── logistic_regression.joblib
├── decision_tree.joblib
├── knn.joblib
├── naive_bayes.joblib
└── svm.joblib
```

It also exports the held-out test set as:

```text
test_data.csv
```

The test set contains **885 rows** and was validated in the notebook to have **0 missing values** and **0 duplicate rows**.
