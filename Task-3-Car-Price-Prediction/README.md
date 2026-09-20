# Car Price Prediction Using Regression

> **Horizon TechX Internship - Task 3**

---

## 1. Project Information
- **Title**: Car Price Prediction Using Regression
- **Company**: Horizon TechX
- **Student**: Tharani Natarajan
- **College**: IFET College of Engineering
- **Department**: Artificial Intelligence and Data Science

## 2. Objective
Build a professional machine learning project that predicts the selling price of a car using relevant vehicle features, applying robust data preprocessing, exploratory data analysis, and predictive modeling using regression algorithms.

## 3. Dataset Information
> **IMPORTANT**: The dataset used in this project was **NOT** supplied by Horizon TechX. 
> It was independently sourced from the **UCI Machine Learning Repository (Automobile Data Set, 1985)** to fulfill the requirements of this task.

- **Source**: UCI Machine Learning Repository
- **Instances**: 205 (201 after cleaning)
- **Features**: 26 (including target 'price')
- **Description**: Contains specifications of various automobiles (e.g., engine size, horsepower, dimensions) and their assigned insurance risk rating, normalized losses, and selling price.

## 4. Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Machine learning pipelines, models, and evaluation
- **Jupyter Notebook**: Interactive analysis
- **Joblib**: Model serialization

## 5. Project Structure
```text
Task-3-Car-Price-Prediction/
├── README.md                           ← Project documentation
├── requirements.txt                    ← Required Python packages
├── .gitignore                          ← Git ignore rules
├── data/
│   ├── README.md                       ← Dataset documentation
│   └── car_price_dataset.csv           ← Raw dataset
├── notebooks/
│   └── car_price_prediction.ipynb      ← Interactive Jupyter Notebook
├── src/
│   └── car_price_prediction.py         ← Main Python script (modular)
├── outputs/
│   ├── figures/                        ← Generated visualizations (300 DPI)
│   └── reports/                        ← Model evaluation metrics and findings
└── models/
    └── car_price_model.pkl             ← Saved model pipeline
```

## 6. Methodology
### A. Data Preprocessing
- Missing values in numerical features (like horsepower) imputed with the median.
- Missing values in categorical features imputed with the most frequent value.
- Numerical features standardized using `StandardScaler`.
- Categorical features encoded using `OneHotEncoder`.

### B. Feature Engineering
- Extracted combined fuel efficiency feature (`avg-mpg`) from city and highway MPG.
- Target variable (`price`) was analyzed for skewness.

### C. Models Trained
1. **Linear Regression** (Baseline)
2. **Random Forest Regressor** (Ensemble)
3. **Gradient Boosting Regressor** (Ensemble)

## 7. Model Evaluation
| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| Linear Regression | 1983.81 | 10704030.89 | 3271.70 | 0.9125 |
| Random Forest | 1793.17 | 7972081.88 | 2823.49 | 0.9348 |
| **Gradient Boosting** | **1581.85** | **6446785.37** | **2539.05** | **0.9473** |

*Gradient Boosting achieved the highest R² score and lowest error metrics.*

## 8. Key Findings
- **Feature Importance**: Engine size, curb weight, and horsepower are the strongest predictors of a car's price.
- **Model Choice**: Tree-based ensemble models significantly outperform standard linear regression due to their ability to capture non-linear relationships.
- **Fuel Efficiency**: There is a strong negative correlation between MPG and price; luxury/expensive cars tend to have lower fuel efficiency.

## 9. How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main pipeline script:
   ```bash
   python src/car_price_prediction.py
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/car_price_prediction.ipynb
   ```

## 10. Sample Prediction Demonstration
To predict a car's price using the saved model:
```python
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load('models/car_price_model.pkl')

# Create a sample input (must match the feature columns)
sample_data = pd.DataFrame([{
    'wheel-base': 99.4, 'length': 176.6, 'width': 66.4, 'height': 54.3, 
    'curb-weight': 2824, 'engine-size': 136, 'bore': 3.19, 'stroke': 3.40, 
    'compression-ratio': 8.0, 'horsepower': 115, 'peak-rpm': 5500, 'avg-mpg': 20.0,
    'make': 'audi', 'fuel-type': 'gas', 'aspiration': 'std', 'num-of-doors': 'four', 
    'body-style': 'sedan', 'drive-wheels': 'fwd', 'engine-location': 'front', 
    'engine-type': 'ohc', 'num-of-cylinders': 'five', 'fuel-system': 'mpfi'
}])

# Predict
predicted_price = model.predict(sample_data)
print(f"Predicted Price: ${predicted_price[0]:.2f}")
```

## 11. Limitations
- The dataset is relatively small (205 instances), which limits the model's ability to generalize to unseen modern car variants.
- Feature importance analysis is dependent on the specific subset of cars represented in this 1985 dataset.
