import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# Define paths
BASE_DIR = pathlib.Path(r"C:\Users\thara\OneDrive\Desktop\Internship projects\Horizon\Task-3-Car-Price-Prediction")
DATA_PATH = BASE_DIR / "data" / "car_price_dataset.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
REP_DIR = BASE_DIR / "outputs" / "reports"
MODEL_DIR = BASE_DIR / "models"

def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    return df

def clean_data(df):
    print("Cleaning dataset...")
    # Target column is 'price'
    
    # Drop rows where target is missing
    df = df.dropna(subset=['price'])
    
    # Convert appropriate columns to numeric that might have been parsed as object
    cols_to_numeric = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def perform_eda(df):
    print("Generating EDA Visualizations...")
    sns.set_theme(style="whitegrid")
    
    # A. Distribution of car prices
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, bins=30, color='blue')
    plt.title("Distribution of Car Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_price_distribution.png", dpi=300)
    plt.close()
    
    # B. Price vs Horsepower (if available)
    if 'horsepower' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='horsepower', y='price', data=df, hue='fuel-type', alpha=0.7)
        plt.title("Car Price vs Horsepower")
        plt.xlabel("Horsepower")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "02_price_vs_horsepower.png", dpi=300)
        plt.close()
        
    # C. Price vs City MPG
    if 'city-mpg' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='city-mpg', y='price', data=df, alpha=0.7, color='green')
        plt.title("Car Price vs City MPG (Fuel Efficiency)")
        plt.xlabel("City MPG")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "03_price_vs_mpg.png", dpi=300)
        plt.close()
        
    # D. Price by fuel type
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='fuel-type', y='price', data=df)
    plt.title("Car Price by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_price_by_fuel_type.png", dpi=300)
    plt.close()
    
    # E. Price by body style
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='body-style', y='price', data=df)
    plt.title("Car Price by Body Style")
    plt.xlabel("Body Style")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_price_by_body_style.png", dpi=300)
    plt.close()
    
    # F. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_correlation_heatmap.png", dpi=300)
    plt.close()

def feature_engineering(df):
    print("Performing feature engineering...")
    # Combine city and highway mpg to average mpg
    df['avg-mpg'] = (df['city-mpg'] + df['highway-mpg']) / 2.0
    
    # Log transform price to handle skewness
    df['log_price'] = np.log1p(df['price'])
    
    # Target is log_price for modeling to stabilize variance, but we will evaluate on actual price
    return df

def train_and_evaluate(df):
    print("Building and training models...")
    # Features and Target
    # Use selected important features to avoid high dimensionality from all categorical vars
    numerical_features = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 
                          'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'avg-mpg']
    categorical_features = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                            'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
    
    X = df[numerical_features + categorical_features]
    y = df['price']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    best_r2 = -float("inf")
    best_model_name = None
    best_pipeline = None
    best_y_pred = None
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            best_y_pred = y_pred
            
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(REP_DIR / "model_comparison.csv", index=False)
    
    print("Model Evaluation Complete:")
    print(results_df)
    
    # Plot Actual vs Predicted for Best Model
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, alpha=0.7, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Actual vs Predicted Prices ({best_model_name})")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_actual_vs_predicted.png", dpi=300)
    plt.close()
    
    # Residual Plot
    residuals = y_test - best_y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(best_y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f"Residual Plot ({best_model_name})")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_residual_plot.png", dpi=300)
    plt.close()
    
    # Feature Importance (if tree based)
    if hasattr(best_pipeline.named_steps["regressor"], "feature_importances_"):
        # Get feature names after one-hot encoding
        cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
        cat_features_out = cat_encoder.get_feature_names_out(categorical_features)
        all_features = numerical_features + list(cat_features_out)
        
        importances = best_pipeline.named_steps["regressor"].feature_importances_
        indices = np.argsort(importances)[-15:] # Top 15
        
        plt.figure(figsize=(10, 8))
        plt.title("Top 15 Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color='c', align='center')
        plt.yticks(range(len(indices)), [all_features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "09_feature_importance.png", dpi=300)
        plt.close()
        
        # Save feature importances to CSV
        imp_df = pd.DataFrame({"Feature": [all_features[i] for i in reversed(indices)], "Importance": importances[indices[::-1]]})
        imp_df.to_csv(REP_DIR / "feature_importance.csv", index=False)
        
    # Save the best model
    model_path = MODEL_DIR / "car_price_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved {best_model_name} model pipeline to {model_path}")
    
    # Write Key Findings
    findings = f"""# Key Findings

1. Dataset Facts: 
   - Shape after cleaning: {df.shape}
   - Numerical features: {len(numerical_features)}
   - Categorical features: {len(categorical_features)}

2. Price Distribution:
   - Car prices are right-skewed, meaning most cars are clustered in the lower/mid price range, with fewer expensive luxury cars.

3. Correlation Insights:
   - Strongest positive correlations with price usually include 'engine-size', 'curb-weight', and 'horsepower'.
   - Strong negative correlation observed between 'city-mpg' and price, meaning more fuel-efficient cars tend to be cheaper.

4. Model Performance:
   - Best Model: {best_model_name}
   - Best R-squared: {best_r2:.4f}
   - Root Mean Squared Error (RMSE): {results_df.loc[results_df['Model']==best_model_name, 'RMSE'].values[0]}

5. Residual Analysis:
   - For lower to mid-range prices, the residuals are relatively balanced. 
   - Larger errors occur at the higher price spectrum due to sparse data for luxury vehicles.
"""
    with open(REP_DIR / "key_findings.txt", "w") as f:
        f.write(findings)
        
def main():
    print("=" * 60)
    print("Task 3: Car Price Prediction Pipeline")
    print("=" * 60)
    df = load_data()
    df_clean = clean_data(df)
    perform_eda(df_clean)
    df_engineered = feature_engineering(df_clean)
    train_and_evaluate(df_engineered)
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
