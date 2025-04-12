import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Define the load_and_preprocess_data function here
def load_and_preprocess_data(features_path, targets_path, test_size=0.2, random_state=42):
    """
    Load and preprocess the data for the CNN model
    """
    # Load data
    features_df = pd.read_csv(features_path)
    targets_df = pd.read_csv(targets_path)

    # Merge on profile_id
    data = pd.merge(features_df, targets_df, on='profile_id')

    # Drop profile_id for modeling
    profile_ids = data['profile_id']
    data = data.drop(columns=['profile_id'])

    # Extract target variables
    target_columns = ['stocks_allocation', 'bonds_allocation', 'real_estate_allocation',
                      'commodities_allocation', 'crypto_allocation', 'cash_allocation']
    risk_level = data['risk_level']
    y_allocation = data[target_columns].values

    # One-hot encode risk_level
    encoder = OneHotEncoder(sparse_output=False)
    risk_levels = data['risk_level'].values.reshape(-1, 1)
    y_risk = encoder.fit_transform(risk_levels)

    # Remove target columns from features
    X = data.drop(columns=target_columns + ['risk_level'])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_alloc_train, y_alloc_test, y_risk_train, y_risk_test = train_test_split(
        X_scaled, y_allocation, y_risk, test_size=test_size, random_state=random_state)

    # Reshape for CNN (add channel dimension)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return {
        'X_train': X_train_reshaped,
        'X_test': X_test_reshaped,
        'y_alloc_train': y_alloc_train,
        'y_alloc_test': y_alloc_test,
        'y_risk_train': y_risk_train,
        'y_risk_test': y_risk_test,
        'feature_names': X.columns.tolist(),
        'target_columns': target_columns,
        'risk_levels': encoder.categories_[0],
        'scaler': scaler,
        'profile_ids': profile_ids
    }

# Load the trained model
try:
    model = tf.keras.models.load_model('best_investment_cnn_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Load feature names, scaler, and risk_levels
try:
    data = load_and_preprocess_data(
        features_path='cnn_features.csv',
        targets_path='cnn_targets.csv'
    )
    feature_names = data['feature_names']
    scaler = data['scaler']
    risk_levels = data['risk_levels']
    print("Features, scaler, and risk levels loaded successfully.")
except Exception as e:
    print(f"Error loading features, scaler, or risk levels: {e}")
    exit()

def preprocess_user_data(user_data, feature_names, scaler):
    """
    Preprocesses the user data using the provided scaler and feature names.
    """
    # Convert user_data to a Pandas Series to use feature names
    user_series = pd.Series(user_data, index=feature_names)
    # Convert the Pandas Series to a DataFrame
    user_df = pd.DataFrame([user_series])
    # Transform the user data using the scaler
    user_scaled = scaler.transform(user_df)
    return user_scaled

def make_recommendation(model, user_data_scaled, risk_levels):
    """
    Makes investment recommendations based on preprocessed user data.
    """
    # Reshape user data for CNN input
    user_features_reshaped = user_data_scaled.reshape(1, user_data_scaled.shape[1], 1)
    # Generate predictions
    pred_allocation, pred_risk = model.predict(user_features_reshaped)

    # Extract allocations
    allocation_dict = {
        'Stocks': pred_allocation[0][0],
        'Bonds': pred_allocation[0][1],
        'Real Estate': pred_allocation[0][2],
        'Commodities': pred_allocation[0][3],
        'Crypto': pred_allocation[0][4],
        'Cash': pred_allocation[0][5]
    }

    # Determine risk level
    risk_index = np.argmax(pred_risk[0])
    risk_level = risk_levels[risk_index]

    # Generate recommendation text
    recommendation = f"Recommended Investment Strategy: {risk_level} Risk\n\n"
    recommendation += "Portfolio Allocation:\n"
    for asset, allocation in allocation_dict.items():
        recommendation += f"- {asset}: {allocation*100:.1f}%\n"

    # Investment strategy explanation
    if risk_level in ['Very Low', 'Low']:
        recommendation += "\nThis conservative strategy prioritizes capital preservation with a focus on bonds and cash. "
        recommendation += "Suitable for shorter investment horizons or high liquidity needs."
    elif risk_level in ['Medium']:
        recommendation += "\nThis balanced strategy aims for moderate growth while managing volatility. "
        recommendation += "Good for medium-term investment horizons with a mix of growth and income."
    else:
        recommendation += "\nThis growth-oriented strategy focuses on capital appreciation through higher equity exposure. "
        recommendation += "Best for longer investment horizons where short-term volatility can be tolerated."
    return recommendation

# Example usage
# Ensure this matches the number and order of features used during training
sample_user_data = np.array([0.9, 25000, 1, 3, 0.2, 10, 0.8, 0.4, 0.3, 0.2, 0.1, 0.5])

try:
    user_data_scaled = preprocess_user_data(sample_user_data, feature_names, scaler)
    recommendation = make_recommendation(model, user_data_scaled, risk_levels)
    print(recommendation)
except Exception as e:
    print(f"Error during recommendation: {e}")
