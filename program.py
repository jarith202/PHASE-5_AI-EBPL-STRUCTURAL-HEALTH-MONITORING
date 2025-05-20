PHASE-3
___________________________________________________________________________________________________

# AI Powered Structural Health Monitoring (SHM) Example
# This code simulates vibration sensor data for a structure in healthy and damaged conditions.
# A machine learning classifier is trained to detect structural damage based on extracted features.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Simulate sensor data (vibration signals) for healthy and damaged states

def simulate_vibration_data(n_samples=500, n_timesteps=100, damaged=False):
    """
    Simulate vibration signals for structural health monitoring.
    Healthy signals are sine waves with small noise.
    Damaged signals have altered frequency and additional noise.
    """
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        time = np.linspace(0, 1, n_timesteps)
        freq = 50  # base frequency in Hz
        if damaged:
            freq += 5  # damage causes frequency shift
            noise = np.random.normal(0, 0.5, n_timesteps)
        else:
            noise = np.random.normal(0, 0.2, n_timesteps)
        signal = np.sin(2 * np.pi * freq * time) + noise
        data.append(signal)
    return np.array(data)

# Generate dataset
n_samples = 500
timesteps = 100
healthy_data = simulate_vibration_data(n_samples=n_samples, n_timesteps=timesteps, damaged=False)
damaged_data = simulate_vibration_data(n_samples=n_samples, n_timesteps=timesteps, damaged=True)

# Step 2: Feature extraction
# Extract simple statistical features from each vibration time series

def extract_features(data):
    features = []
    for signal in data:
        feat = [
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            np.max(signal) - np.min(signal),  # peak-to-peak amplitude
        ]
        features.append(feat)
    return np.array(features)

healthy_features = extract_features(healthy_data)
damaged_features = extract_features(damaged_data)

# Combine dataset and create labels
X = np.vstack((healthy_features, damaged_features))
y = np.hstack((np.zeros(len(healthy_features)), np.ones(len(damaged_features))))  # 0 = healthy, 1 = damaged

# Step 3: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train an AI model - Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Visualize example signals
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(healthy_data[0])
plt.title("Example Healthy Signal")
plt.subplot(2,1,2)
plt.plot(damaged_data[0])
plt.title("Example Damaged Signal")
plt.tight_layout()
plt.show()

# End of Code


PHASE-4

___________________________________________________________________________________________________

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Simulate synthetic structural health monitoring (SHM) data
def generate_shm_data(samples=1000):
    np.random.seed(42)

    # Healthy structures
    healthy = np.random.normal(loc=0.5, scale=0.1, size=(samples // 2, 3))

    # Damaged structures
    damaged = np.random.normal(loc=1.1, scale=0.2, size=(samples // 2, 3))

    # Combine the data
    X = np.vstack((healthy, damaged))
    y = np.array([0] * (samples // 2) + [1] * (samples // 2))  # 0 = Healthy, 1 = Damaged

    return pd.DataFrame(X, columns=["Sensor1", "Sensor2", "Sensor3"]), y

# Plot function
def plot_data(X, y):
    plt.figure(figsize=(12, 4))
    sensors = ["Sensor1", "Sensor2", "Sensor3"]

    for i, sensor in enumerate(sensors):
        plt.subplot(1, 3, i + 1)
        plt.scatter(X[sensor][y == 0], X["Sensor3"][y == 0], color="blue", alpha=0.5, label="Healthy")
        plt.scatter(X[sensor][y == 1], X["Sensor3"][y == 1], color="red", alpha=0.5, label="Damaged")
        plt.xlabel(sensor)
        plt.ylabel("Sensor3")
        plt.title(f"{sensor} vs Sensor3")
        if i == 0:
            plt.legend()

    plt.suptitle("Sensor Readings: Healthy vs Damaged")
    plt.tight_layout()
    plt.show()

# Generate and visualize data
X, y = generate_shm_data()
plot_data(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict on new data
new_data = np.array([[0.6, 0.55, 0.58], [1.2, 1.0, 0.95]])  # First likely healthy, second likely damaged
predictions = model.predict(new_data)

for i, pred in enumerate(predictions):
    print(f"Sample {i+1} is", "Damaged" if pred else "Healthy")


#End of the code.
___________________________________________________________________________________________________
