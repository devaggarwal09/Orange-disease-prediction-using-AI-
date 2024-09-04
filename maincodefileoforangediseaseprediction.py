# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 10:51:18 2024

@author: Dev
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set the paths to your train and test directories
train_dir = "orange/dataset/train"
test_dir = "orange/dataset/test"

# Set parameters
batch_size = 32
img_height = 150
img_width = 150
epochs = 5

# Preprocess and augment the data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)  # Do not shuffle for evaluation

# Get class labels
class_labels = list(train_data.class_indices.keys())

# Load the ResNet50 model without the top fully connected layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(class_labels), activation='softmax')(x)  # Dynamic based on number of classes

# Create the final model
cnn_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(train_data, epochs=epochs, validation_data=test_data)  # Add validation data

# Use the CNN model for feature extraction
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)

# Function to extract features
def extract_features(model, data, steps):
    features = []
    labels = []
    for i in range(steps):
        images, label_batch = next(data)
        features_batch = model.predict(images)
        features.append(features_batch)
        labels.append(label_batch)
    features = np.vstack(features)
    labels = np.vstack(labels)
    return features, labels

# Calculate steps per epoch
train_steps = train_data.samples // batch_size
test_steps = test_data.samples // batch_size

# Extract features from the training and test data
train_features, train_labels = extract_features(feature_extractor, train_data, train_steps)
test_features, test_labels = extract_features(feature_extractor, test_data, test_steps)

# Flatten the labels to fit the GBM classifier
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Train the GBM classifier
gbm_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
gbm_classifier.fit(train_features, train_labels)

# Make predictions on the test data
test_predictions = gbm_classifier.predict(test_features)

# Calculate metrics
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average='weighted')
recall = recall_score(test_labels, test_predictions, average='weighted')
f1 = f1_score(test_labels, test_predictions, average='weighted')

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
print(f'Confusion Matrix:\n{conf_matrix}')

# Calculate TP, TN, FP, FN for each class
tp_per_class = np.diag(conf_matrix)
fp_per_class = conf_matrix.sum(axis=0) - tp_per_class
fn_per_class = conf_matrix.sum(axis=1) - tp_per_class
tn_per_class = conf_matrix.sum() - (fp_per_class + fn_per_class + tp_per_class)

# Aggregate TP, TN, FP, FN if needed
tp = tp_per_class.sum()
fp = fp_per_class.sum()
fn = fn_per_class.sum()
tn = tn_per_class.sum()

# Print metrics
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'True Positives: {tp}')
print(f'True Negatives: {tn}')
print(f'False Positives: {fp}')
print(f'False Negatives: {fn}')

# Map predictions to disease names
predicted_disease_names = [class_labels[i] for i in test_predictions]

# Display some predictions
for i in range(min(10, len(predicted_disease_names))):  # Display first 10 predictions
    print(f'Predicted Disease: {predicted_disease_names[i]}, True Label: {class_labels[test_labels[i]]}')