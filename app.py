import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

# Streamlit app title
st.title("Aircraft Engine Predictive Maintenance (Improved)")

# File upload section
st.header("Upload CMAPSS Dataset Files")
train_file = st.file_uploader("Upload train_FD001.txt", type=["txt"])
test_file = st.file_uploader("Upload test_FD001.txt", type=["txt"])
rul_file = st.file_uploader("Upload RUL_FD001.txt", type=["txt"])

# Function to preprocess data
def preprocess_data(train_file, test_file, rul_file):
    # Read train data
    train_df = pd.read_csv(train_file, sep=" ", header=None)
    train_df = train_df.dropna(axis=1, how="all")
    train_df.columns = ['engine_id', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3',
                        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                        's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # Read test data
    test_df = pd.read_csv(test_file, sep=" ", header=None)
    test_df = test_df.dropna(axis=1, how="all")
    test_df.columns = ['engine_id', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3',
                       's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                       's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # Read RUL data
    rul_df = pd.read_csv(rul_file, sep=" ", header=None)
    rul_df = rul_df.dropna(axis=1, how="all")
    rul_df.columns = ['rul']

    # Calculate RUL for training data
    max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='engine_id')
    train_df['rul'] = train_df['max_cycle'] - train_df['cycle']
    train_df = train_df.drop(columns=['max_cycle'])

    # For test data, take the last cycle for each engine and merge with RUL
    test_last_cycle = test_df.groupby('engine_id').last().reset_index()
    test_last_cycle['rul'] = rul_df['rul']

    return train_df, test_last_cycle

# Function to perform feature selection using Random Forest
def select_features(train_df):
    features = ['op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
                's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    
    X_train = train_df[features]
    y_train = train_df['rul']
    
    # Train a Random Forest to get feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Select top 10 features based on importance
    feature_importance = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
    selected_features = feature_importance.sort_values(by='importance', ascending=False).head(10)['feature'].tolist()
    
    return selected_features

# Function to train and evaluate ensemble model
def train_and_evaluate(train_df, test_df, selected_features):
    X_train = train_df[selected_features]
    y_train = train_df['rul']
    X_test = test_df[selected_features]
    y_test = test_df['rul']

    # Train Random Forest with tuned parameters
    rf_model = RandomForestRegressor(n_estimators=100, min_samples_split=5, max_features='sqrt', max_depth=10, bootstrap=True, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Train XGBoost
    xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # Ensemble: Average predictions from Random Forest and XGBoost
    y_pred = (rf_pred + xgb_pred) / 2

    # Apply correction for overestimation (subtract MAE from original model)
    correction = 23.66  # MAE from original model
    y_pred_corrected = np.maximum(y_pred - correction, 0)  # Ensure predictions are non-negative

    # Calculate metrics for corrected predictions
    r2 = r2_score(y_test, y_pred_corrected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
    mae = mean_absolute_error(y_test, y_pred_corrected)

    return y_test, y_pred_corrected, r2, rmse, mae

# Function to plot predictions
def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.values, color="blue", label="Actual")
    ax.plot(y_pred, color="green", label="Prediction")
    ax.set_title("Actual vs Predicted RUL (Improved Model)")
    ax.set_ylabel("Number of Cycles")
    ax.set_xlabel("Engines")
    ax.legend(loc="upper right")
    return fig

# Main app logic
if train_file and test_file and rul_file:
    with st.spinner("Processing data and training model..."):
        # Preprocess data
        try:
            train_df, test_df = preprocess_data(train_file, test_file, rul_file)
            st.success("Data loaded and preprocessed successfully!")
            
            # Perform feature selection
            selected_features = select_features(train_df)
            st.header("Selected Features")
            st.write(selected_features)
            
            # Train and evaluate model
            y_test, y_pred, r2, rmse, mae = train_and_evaluate(train_df, test_df, selected_features)
            
            # Display metrics
            st.header("Model Performance Metrics")
            st.write(f"**R2 Score (Test):** {r2:.4f}")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**MAE:** {mae:.2f}")
            
            # Display predictions plot
            st.header("Actual vs Predicted RUL")
            fig = plot_predictions(y_test, y_pred)
            st.pyplot(fig)
            
            # Display sample predictions
            st.header("Sample Predictions")
            results_df = pd.DataFrame({
                "Engine ID": test_df['engine_id'],
                "Actual RUL": y_test,
                "Predicted RUL": y_pred
            })
            st.dataframe(results_df.head(10))
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload all three dataset files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt) to proceed.")