import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Ensure reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(3407)
np.random.seed(3407)
tf.random.set_seed(3407)

# Load dataset
input_file = "dataProcessed/BOU2019_HA.csv"
df = pd.read_csv(input_file, delimiter=',')

# Configuration
total_train_points = 31964
look_back_points = 100
total_test_points = 2976
start_index = 0
utide_factor = 1
future_steps_list = [25]

# Check dataset size
total_required = total_train_points + look_back_points + total_test_points
if len(df) < total_required + start_index:
    raise ValueError("Dataset is too small to meet the requirement.")

# Slice the dataset and reset index
df_sampled = df.iloc[start_index:start_index + total_required].reset_index(drop=True)

# Extract anomaly and tide_h columns
anomaly_vals = df_sampled['anomaly'].values.reshape(-1, 1)
tide_vals = df_sampled['tide_h'].values.reshape(-1, 1) * utide_factor

# Fit scalers
scaler_anomaly = MinMaxScaler(feature_range=(0, 1))
scaler_tide = MinMaxScaler(feature_range=(0, 1))
scaler_anomaly.fit(anomaly_vals)
scaler_tide.fit(tide_vals)

# Scale the data
anomaly_scaled = scaler_anomaly.transform(anomaly_vals)
tide_scaled = scaler_tide.transform(tide_vals)
dataset_scaled_multi = np.hstack((anomaly_scaled, tide_scaled))

# Split into training set and test set
train_multi = dataset_scaled_multi[:total_train_points]
test_multi = dataset_scaled_multi[total_train_points : total_train_points + total_test_points + look_back_points]

def create_dataset_multi_multioutput(data, look_back, future_steps):
    """
    Create a multi-output dataset for sequence-to-sequence prediction.
    Each sample has `look_back` timesteps as input,
    and `future_steps` timesteps of future anomaly values as output.
    """
    dataX, dataY = [], []
    for i in range(len(data) - look_back - future_steps + 1):
        seq_x = data[i : i + look_back, :]
        seq_y = [data[i + look_back + j, 0] for j in range(future_steps)]  # future anomaly values
        dataX.append(seq_x)
        dataY.append(seq_y)
    return np.array(dataX), np.array(dataY)

def inverse_anomaly_array(arr):
    """
    Inverse transform for anomaly data.
    """
    arr = arr.reshape(-1, 1)
    return scaler_anomaly.inverse_transform(arr).flatten()

def inverse_tide_array_factor(arr):
    """
    Inverse transform for tide data and divide by the utide factor.
    """
    arr = arr.reshape(-1, 1)
    tide_inversed = scaler_tide.inverse_transform(arr).flatten()
    return tide_inversed / utide_factor

def build_bilstm_multi(input_shape, future_steps):
    """
    Build a BiLSTM model for multi-output regression.
    """
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.3)(x)
    # x = Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.0001)))(x)
    # x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(100, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(future_steps)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

results_to_save = []

for future_steps in future_steps_list:
    # Prepare training and test data
    trainX_multi, trainY_multi = create_dataset_multi_multioutput(train_multi, look_back_points, future_steps)
    testX_multi, testY_multi = create_dataset_multi_multioutput(test_multi, look_back_points, future_steps)

    # Split training data into training and validation sets
    trainX, valX, trainY, valY = train_test_split(trainX_multi, trainY_multi, test_size=0.1, random_state=3407)

    # Build and train the model
    model = build_bilstm_multi((look_back_points, 2), future_steps)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
    
    print(f"Training model (utide_factor={utide_factor}, future_steps={future_steps})...")
    model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=80,
        batch_size=256,
        verbose=1,
        callbacks=[reduce_lr]
    )
    print("Training completed.\n")

    # Perform rolling prediction
    current_input = test_multi[:look_back_points].copy()
    predictions_scaled = []
    actual_scaled = []
    tide_scaled_arr = []

    predicted_points = 0
    while predicted_points < total_test_points:
        input_sequence = current_input.reshape(1, look_back_points, 2)
        future_pred_scaled = model.predict(input_sequence, verbose=0)[0]

        for step_i in range(future_steps):
            if predicted_points >= total_test_points:
                break
            predictions_scaled.append(future_pred_scaled[step_i])
            actual_scaled_val = test_multi[look_back_points + predicted_points, 0]
            tide_scaled_val = test_multi[look_back_points + predicted_points, 1]
            actual_scaled.append(actual_scaled_val)
            tide_scaled_arr.append(tide_scaled_val)

            # Shift the input window by one timestep
            current_input = np.roll(current_input, -1, axis=0)
            current_input[-1, 0] = future_pred_scaled[step_i]
            if predicted_points + 1 < total_test_points:
                current_input[-1, 1] = test_multi[look_back_points + predicted_points + 1, 1]
            else:
                current_input[-1, 1] = test_multi[look_back_points + predicted_points, 1]

            predicted_points += 1

    # Inverse transform predictions and ground truth
    predicted_anomaly = inverse_anomaly_array(np.array(predictions_scaled))
    actual_anomaly = inverse_anomaly_array(np.array(actual_scaled))
    actual_tide = inverse_tide_array_factor(np.array(tide_scaled_arr))

    # Calculate residuals for the first 1000 points
    residuals_predicted = actual_anomaly[:1000] - predicted_anomaly[:1000]
    residuals_tide = actual_anomaly[:1000] - actual_tide[:1000]

    # Plot actual vs. predicted (first 1000 points)
    plt.figure(figsize=(12, 6))
    plt.plot(actual_anomaly[:1000], label='Actual', color='black')
    plt.plot(predicted_anomaly[:1000], label='Predicted', color='red', linestyle='--')
    plt.plot(actual_tide[:1000], label='Utide', color='green', linestyle='--')
    plt.title(f'Future Steps: {future_steps}')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot residuals
    plt.figure(figsize=(12, 6))
    plt.plot(residuals_predicted, label='Residuals (Actual - Predicted)', color='blue')
    plt.plot(residuals_tide, label='Residuals (Actual - Utide)', color='green', linestyle='--')
    plt.title(f'Residuals (Future Steps: {future_steps})')
    plt.xlabel('Time Step')
    plt.ylabel('Residuals')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.show()

    num_segments = 9
    segment_length = 100
    segment_labels = []
    r2_pred_list, mse_pred_list, mae_pred_list = [], [], []
    r2_tide_list, mse_tide_list, mae_tide_list = [], [], []

    for seg_i in range(num_segments):
        start = seg_i * segment_length
        end = start + segment_length

        seg_actual = actual_anomaly[start:end]
        seg_pred = predicted_anomaly[start:end]
        seg_tide = actual_tide[start:end]

        r2_pred = r2_score(seg_actual, seg_pred)
        mse_pred = mean_squared_error(seg_actual, seg_pred)
        mae_pred = mean_absolute_error(seg_actual, seg_pred)

        r2_tide = r2_score(seg_actual, seg_tide)
        mse_tide = mean_squared_error(seg_actual, seg_tide)
        mae_tide = mean_absolute_error(seg_actual, seg_tide)

        r2_pred_list.append(r2_pred)
        mse_pred_list.append(mse_pred)
        mae_pred_list.append(mae_pred)

        r2_tide_list.append(r2_tide)
        mse_tide_list.append(mse_tide)
        mae_tide_list.append(mae_tide)

        segment_labels.append(f'{start}-{end}')

    # R²
    x_indices = np.arange(num_segments)
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices, r2_pred_list, marker='o', color='red', label='Predicted R²')
    plt.plot(x_indices, r2_tide_list, marker='o', color='green', label='Utide R²')
    plt.title(f'R² per 100-step segment (Future Steps: {future_steps})')
    plt.xticks(x_indices, segment_labels, rotation=45)
    plt.ylabel('R²')
    plt.grid(True, linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # MSE
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices, mse_pred_list, marker='o', color='red', label='Predicted MSE')
    plt.plot(x_indices, mse_tide_list, marker='o', color='green', label='Utide MSE')
    plt.title(f'MSE per 100-step segment (Future Steps: {future_steps})')
    plt.xticks(x_indices, segment_labels, rotation=45)
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # MAE
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices, mae_pred_list, marker='o', color='red', label='Predicted MAE')
    plt.plot(x_indices, mae_tide_list, marker='o', color='green', label='Utide MAE')
    plt.title(f'MAE per 100-step segment (Future Steps: {future_steps})')
    plt.xticks(x_indices, segment_labels, rotation=45)
    plt.ylabel('MAE')
    plt.grid(True, linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for t, (pred_val, actual_val, tide_val) in enumerate(zip(predicted_anomaly, actual_anomaly, actual_tide)):
        results_to_save.append({
            'future_steps': future_steps,
            'Time_Step': t,
            'Predicted_Values': pred_val,
            'Actual_Values': actual_val,
            'Utide_Values': tide_val
        })
results_df = pd.DataFrame(results_to_save)
