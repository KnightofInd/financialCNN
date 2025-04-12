import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Preprocessing
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

# 2. Build CNN Model
def build_cnn_model(input_shape, num_risk_levels):
    """
    Build a CNN model for predicting portfolio allocations and risk levels
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Flatten before dense layers
    x = layers.Flatten()(x)

    # Dense layers for feature extraction
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Two output branches: allocation percentages and risk level
    # 1. Portfolio allocation (6 asset classes, must sum to 1)
    allocation_output = layers.Dense(6, activation='softmax', name='allocation_output')(x)

    # 2. Risk level classification
    risk_output = layers.Dense(num_risk_levels, activation='softmax', name='risk_output')(x)

    # Create model with multiple outputs
    model = models.Model(inputs=inputs, outputs=[allocation_output, risk_output])

    # Compile model with appropriate losses
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            'allocation_output': 'mse',  # Mean squared error for allocations
            'risk_output': 'categorical_crossentropy'  # Cross-entropy for risk classification
        },
        loss_weights={
            'allocation_output': 0.7,  # Weight allocation prediction higher
            'risk_output': 0.3
        },
        metrics={
            'allocation_output': 'mae',  # Mean absolute error
            'risk_output': 'accuracy'  # Classification accuracy
        }
    )

    return model

# 3. Train and Evaluate Model
def train_model(model, data, epochs=50, batch_size=32, patience=10, verbose=1):
    """
    Train the CNN model with early stopping
    """
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        'best_investment_cnn_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train model
    history = model.fit(
        data['X_train'],
        {
            'allocation_output': data['y_alloc_train'],
            'risk_output': data['y_risk_train']
        },
        validation_data=(
            data['X_test'],
            {
                'allocation_output': data['y_alloc_test'],
                'risk_output': data['y_risk_test']
            }
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=verbose
    )

    return model, history

# 4. Visualization and Analysis
def visualize_results(model, data, history):
    """
    Visualize the training history and model predictions
    """
    # Plot training history
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['allocation_output_loss'], label='Train')
    plt.plot(history.history['val_allocation_output_loss'], label='Validation')
    plt.title('Allocation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history['risk_output_accuracy'], label='Train')
    plt.plot(history.history['val_risk_output_accuracy'], label='Validation')
    plt.title('Risk Level Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history.history['allocation_output_mae'], label='Train')
    plt.plot(history.history['val_allocation_output_mae'], label='Validation')
    plt.title('Allocation Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history.history['loss'], label='Train Total')
    plt.plot(history.history['val_loss'], label='Validation Total')
    plt.title('Combined Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    # Predict on test data
    y_pred_alloc, y_pred_risk = model.predict(data['X_test'])

    # Compare actual vs predicted allocations
    asset_classes = data['target_columns']

    # Plot actual vs predicted for a few examples
    num_examples = min(5, len(data['X_test']))

    plt.figure(figsize=(15, 10))
    for i in range(num_examples):
        plt.subplot(num_examples, 2, 2*i+1)
        plt.bar(asset_classes, data['y_alloc_test'][i], alpha=0.7, label='Actual')
        plt.bar(asset_classes, y_pred_alloc[i], alpha=0.5, label='Predicted')
        plt.title(f'Example {i+1} Portfolio Allocation')
        plt.ylabel('Allocation %')
        plt.xticks(rotation=45)
        plt.legend()

        plt.subplot(num_examples, 2, 2*i+2)
        risk_levels = data['risk_levels']
        actual_risk = np.argmax(data['y_risk_test'][i])
        pred_risk = np.argmax(y_pred_risk[i])

        risk_comparison = pd.DataFrame({
            'Actual': np.zeros(len(risk_levels)),
            'Predicted': np.zeros(len(risk_levels))
        }, index=risk_levels)
        risk_comparison.loc[risk_levels[actual_risk], 'Actual'] = 1
        risk_comparison.loc[risk_levels[pred_risk], 'Predicted'] = 1
        risk_comparison.plot(kind='bar', ax=plt.gca())
        plt.title(f'Example {i+1} Risk Level')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.close()

    # Allocation error analysis
    mae_per_asset = np.mean(np.abs(y_pred_alloc - data['y_alloc_test']), axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(asset_classes, mae_per_asset)
    plt.title('Mean Absolute Error by Asset Class')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('mae_by_asset.png')
    plt.close()

    # Risk classification confusion matrix
    y_risk_true = np.argmax(data['y_risk_test'], axis=1)
    y_risk_pred = np.argmax(y_pred_risk, axis=1)
    conf_matrix = tf.math.confusion_matrix(y_risk_true, y_risk_pred).numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=data['risk_levels'],
                yticklabels=data['risk_levels'])
    plt.title('Risk Level Confusion Matrix')
    plt.ylabel('True Risk Level')
    plt.xlabel('Predicted Risk Level')
    plt.tight_layout()
    plt.savefig('risk_confusion_matrix.png')
    plt.close()

# 5. Feature Importance Analysis
def analyze_feature_importance(model, data):
    """
    Analyze feature importance using a permutation approach
    """
    feature_names = data['feature_names']
    X_test = data['X_test']
    y_alloc_test = data['y_alloc_test']
    y_risk_test = data['y_risk_test']

    # Get baseline performance
    baseline_results = model.evaluate(
        X_test,
        {
            'allocation_output': y_alloc_test,
            'risk_output': y_risk_test
        },
        verbose=0
    )

    baseline_alloc_mae = baseline_results[3]  # MAE for allocation output
    baseline_risk_acc = baseline_results[4]  # Accuracy for risk output

    # Calculate importance for each feature
    allocation_importance = []
    risk_importance = []
    for i in range(len(feature_names)):
        # Create a copy of the test data
        X_test_permuted = X_test.copy()

        # Permute the feature
        permuted_feature = np.random.permutation(X_test_permuted[:, i, 0])
        X_test_permuted[:, i, 0] = permuted_feature

        # Evaluate on permuted data
        permuted_results = model.evaluate(
            X_test_permuted,
            {
                'allocation_output': y_alloc_test,
                'risk_output': y_risk_test
            },
            verbose=0
        )

        # Calculate importance (increase in error after permutation)
        allocation_importance.append(permuted_results[3] - baseline_alloc_mae)
        risk_importance.append(baseline_risk_acc - permuted_results[4])

    # Normalize importance scores
    allocation_importance = np.array(allocation_importance)
    risk_importance = np.array(risk_importance)
    allocation_importance = allocation_importance / np.sum(np.abs(allocation_importance))
    risk_importance = risk_importance / np.sum(np.abs(risk_importance))

    # Plot importance
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': allocation_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance for Portfolio Allocation')
    plt.xlabel('Normalized Importance')

    plt.subplot(2, 1, 2)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': risk_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance for Risk Level Classification')
    plt.xlabel('Normalized Importance')

    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return {
        'allocation_importance': allocation_importance,
        'risk_importance': risk_importance,
        'feature_names': feature_names
    }

# 6. Recommendation System
def recommend_investment_strategy(model, user_data, scaler, risk_levels, feature_names):
    """
    Generate investment recommendations based on user data
    """
    # Create a DataFrame from user_data with feature names
    user_data_df = pd.DataFrame([user_data], columns=feature_names)

    # Transform the user data using the scaler
    user_features = scaler.transform(user_data_df)

    # Reshape the user features for the CNN model
    user_features_reshaped = user_features.reshape(1, user_features.shape[1], 1)

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

    # Extract risk level
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

    return {
        'allocation': allocation_dict,
        'risk_level': risk_level,
        'recommendation': recommendation
    }

# 7. Time Series CNN for Dynamic Recommendations
def build_time_series_cnn(sequence_length, num_features, num_risk_levels):
    """
    Build a CNN model for time series data to provide dynamic investment recommendations
    """
    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features))

    # Convolutional layers
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Two output branches
    allocation_output = layers.Dense(6, activation='softmax', name='allocation_output')(x)
    risk_output = layers.Dense(num_risk_levels, activation='softmax', name='risk_output')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=[allocation_output, risk_output])

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            'allocation_output': 'mse',
            'risk_output': 'categorical_crossentropy'
        },
        loss_weights={
            'allocation_output': 0.7,
            'risk_output': 0.3
        },
        metrics={
            'allocation_output': 'mae',
            'risk_output': 'accuracy'
        }
    )

    return model

# 8. Main function to run the pipeline
def main():
    """
    Run the complete CNN investment strategy recommendation pipeline
    """
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(
        features_path='cnn_features.csv',
        targets_path='cnn_targets.csv'
    )

    print("\nBuilding CNN model...")
    input_shape = (data['X_train'].shape[1], 1)
    num_risk_levels = len(data['risk_levels'])
    model = build_cnn_model(input_shape, num_risk_levels)
    model.summary()

    print("\nTraining CNN model...")
    model, history = train_model(model, data, epochs=50, batch_size=32, patience=10, verbose=1)

    print("\nVisualizing results...")
    visualize_results(model, data, history)

    print("\nAnalyzing feature importance...")
    analyze_feature_importance(model, data)

    # Example of recommendation system usage (assuming you have user data)
    # Create a sample user data (replace with your actual user data)
    sample_user_data = np.array([0.5, 35000, 2, 6, 0.7, 15, 0.3, 0.6, 0.4, 0.2, 0.3, 0.5])  # Example data
    print("\nGenerating investment recommendation...")
    recommendation = recommend_investment_strategy(model, sample_user_data, data['scaler'], data['risk_levels'], data['feature_names'])
    print(recommendation['recommendation'])

if __name__ == "__main__":
    main()
