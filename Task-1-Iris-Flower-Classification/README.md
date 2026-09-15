# Iris Flower Classification – Task 1

## 1. Project Title
**Iris Flower Classification** – A beginner‑friendly end‑to‑end machine‑learning project for the Horizon TechX internship.

## 2. Internship Task Description
Implement a complete supervised‑learning pipeline that loads the classic Iris dataset, explores the data, trains a model, evaluates its performance, and provides a reusable prediction function.

## 3. Problem Statement
Given four morphological measurements of an iris flower (sepal length, sepal width, petal length, petal width), predict the species (`setosa`, `versicolor`, or `virginica`).

## 4. Objectives
- Load and inspect the dataset.
- Perform exploratory data analysis (EDA) with clear visualisations.
- Preprocess the data with proper train‑test splitting and scaling.
- Train a **Logistic Regression** classifier (primary) and a **K‑Nearest Neighbours** classifier (comparison).
- Evaluate models using accuracy, precision, recall, F1‑score, and confusion matrices.
- Demonstrate predictions on new measurements.
- Document the whole process for reproducibility.

## 5. Technologies Used
- **Python 3.10+**
- **pandas**, **numpy** – data manipulation
- **matplotlib**, **seaborn** – visualisation
- **scikit‑learn** – modelling and evaluation
- **Jupyter notebooks** – optional exploratory notebooks (stored in `notebooks/`)

## 6. Dataset Description
The Iris dataset is loaded directly from `sklearn.datasets.load_iris`. It contains 150 samples with the following columns:
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (target label)

## 7. Project Folder Structure
```
Task-1-Iris-Flower-Classification/
│   README.md                # <‑‑ you are reading this file
│   
├── data/                    # (empty – dataset is loaded from scikit‑learn)
│
├── notebooks/               # optional Jupyter notebooks for EDA
│
├── src/
│   └── iris_classification.py  # complete pipeline script
│
└── outputs/
    └── figures/
        ├── class_distribution.png
        ├── pairplot.png
        ├── correlation_heatmap.png
        ├── feature_boxplot.png
        ├── confusion_matrix_logistic_regression.png
        └── confusion_matrix_k-nearest_neighbours.png
```

## 8. Installation Instructions
```bash
# From the repository root
cd Horizon-Technologies
# (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# Install required packages
pip install -r requirements.txt
```

## 9. How to Run the Project
```bash
# From the repository root
python Task-1-Iris-Flower-Classification/src/iris_classification.py
```
The script prints data summaries, saves the figures into `outputs/figures/`, displays model evaluation metrics, and shows a prediction demo.

## 10. Data Preprocessing Steps
1. **Feature/Target split** – `X` contains the four measurements, `y` the species.
2. **Label encoding** – converting string species names to integer indices.
3. **Train‑test split** – 80 % training, 20 % testing, `random_state=42`, stratified.
4. **Standard scaling** – fitted on the training set only to avoid leakage, then applied to both train and test.

## 11. Exploratory Data Analysis (EDA)
- **Class distribution bar chart** – visualises balanced classes.
- **Pairplot** – shows pairwise relationships coloured by species.
- **Correlation heatmap** – numeric feature correlations.
- **Feature boxplot** – distribution of each feature per species.
All figures are saved under `outputs/figures/`.

## 12. Machine‑Learning Algorithm Explanation
- **Logistic Regression** – a linear model that estimates class probabilities using the logistic (sigmoid) function and selects the class with highest probability. It works well for linearly separable data and is easy to interpret.
- **K‑Nearest Neighbours (K‑NN)** – a non‑parametric method that classifies a sample based on the majority label of its *k* closest training points in feature space. It provides a contrasting, instance‑based approach.

## 13. Evaluation Metrics
| Metric | Logistic Regression | K‑Nearest Neighbours |
|--------|---------------------|----------------------|
| Accuracy | **0.9333** | **0.9333** |
| Precision (macro) | 0.93 | 0.94 |
| Recall (macro) | 0.93 | 0.93 |
| F1‑score (macro) | 0.93 | 0.93 |

The full classification reports are printed by the script (see the *Testing Summary* below).

## 14. Actual Results Obtained
```
Logistic Regression Evaluation:
Accuracy: 0.9333
              precision    recall  f1-score   support

    setosa       1.00      1.00      1.00        10
versicolor       0.90      0.90      0.90        10
 virginica       0.90      0.90      0.90        10

  accuracy                           0.93        30
 macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30

K-Nearest Neighbours Evaluation:
Accuracy: 0.9333
              precision    recall  f1-score   support

    setosa       1.00      1.00      1.00        10
versicolor       0.83      1.00      0.91        10
 virginica       1.00      0.80      0.89        10

  accuracy                           0.93        30
 macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30
```
Both models achieve the same overall accuracy of **93.33 %** on the held‑out test set.

## 15. Key Findings
- The dataset is clean (no missing values, only one duplicate row which does not affect results).
- Logistic Regression performs very well, achieving perfect precision/recall for *setosa* and high scores for the other two classes.
- K‑NN gives comparable accuracy but shows slightly lower precision for *versicolor* and *virginica* due to class overlap.
- Visualisations clearly illustrate class separability, especially for *setosa*.

## 16. Limitations
- The Iris dataset is small and low‑dimensional; results may not generalise to larger, noisier datasets.
- Only two classifiers are demonstrated; more complex models could be explored for deeper learning.
- No hyperparameter tuning was performed; default settings were used for simplicity.

## 17. Future Improvements
- Add cross‑validation and hyperparameter search (e.g., grid search for `C` in Logistic Regression).
- Experiment with other algorithms (Decision Trees, SVMs, Random Forests).
- Include model persistence (`joblib.dump`) for re‑using the trained model.
- Extend the project with a small Flask or FastAPI web service for interactive predictions.

## 18. Author
**Name:** *<Your Name Here>*
**Institution:** B.Tech in Artificial Intelligence and Data Science
**Internship:** Horizon TechX
