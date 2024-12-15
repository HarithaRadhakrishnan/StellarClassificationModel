# StellarClassificationModel
#Stellar classification of celestial objects based on photometric filters
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Load the dataset
file_path = '/content/drive/MyDrive/finalsky.csv'  # Update this path to your correct file location

try:
    # Attempt to load the file with the correct delimiter
    data = pd.read_csv(file_path, delimiter=',', header=0, low_memory=False)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Check columns
#print("Columns in the dataset:", data.columns)

# Check if required columns are present
required_columns = ['u', 'g', 'r', 'i', 'z', 'class']
if not all(col in data.columns for col in required_columns):
    print(f"Some required columns ({required_columns}) are missing. Available columns: {list(data.columns)}")
    exit()

# Step 3: Preprocess the dataset
try:
    # Select features (u, g, r, i, z) and target (class)
    X = data[['u', 'g', 'r', 'i', 'z']].apply(pd.to_numeric, errors='coerce')
    y = data['class'].astype(str)  # Ensure the target is treated as a string

    # Drop rows with missing values
    data = data.dropna(subset=['u', 'g', 'r', 'i', 'z', 'class'])
    X = X.dropna()
    y = y[X.index]  # Align target with features
    print("Preprocessing completed successfully.")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()

# Step 4: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Convert labels to integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Step 7: Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(encoder.classes_), activation='softmax'))  # Number of unique classes in 'class'

# Step 8: Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 9: Train the model
epochs = 500  # You can change the number of epochs as needed
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

# Step 10: Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)  # Convert probabilities to class labels
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Save the model
model.save('HR.h5')

# Secoond code block to plot results
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Confusion Matrix
conf_matrix = confusion_matrix(y_new_encoded, y_pred_classes_new)

# Normalize the confusion matrix to display accuracies (percentages)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Calculate overall accuracy
test_accuracy = accuracy_score(y_new_encoded, y_pred_classes_new)

# Plot the confusion matrix and accuracy graph
plt.figure(figsize=(16, 8))

# Subplot 1: Normalized Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix (Normalized)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

# Adjust layout
plt.tight_layout()
plt.show()
