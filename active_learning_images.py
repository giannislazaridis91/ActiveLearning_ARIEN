import json  # For configuration file handling.
import numpy as np  # For numerical operations.
import pandas as pd  # For data handling.
from sklearn.model_selection import train_test_split  # For splitting datasets.
from sklearn.svm import SVC  # Support Vector Classifier for image classification.
from sklearn.metrics import accuracy_score  # For model evaluation.
import joblib  # For saving and loading models.
from PIL import Image  # For image loading and resizing.
import torch  # PyTorch for deep learning operations.
import torch.nn as nn  # For defining the CNN architecture.
import torch.optim as optim  # For optimization.
from torchvision import transforms  # For image transformations.

# Function to load JSON configuration from a given file path.
def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)  # Load and return JSON data as a Python dictionary.

# Function to read and preprocess image files based on annotation configuration.
def read_image_files(annotation_config):
    data = []  # To store the processed images.
    labels = []  # To store the labels.

    # Load paths to images and labels.
    data_path = annotation_config['data']
    labels_path = annotation_config['labels']

    # Read image paths.
    with open(data_path, 'r') as f:
        image_paths = f.read().splitlines()

    # Read corresponding labels.
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()

    # Load and preprocess images.
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB (3 channels).
        image = image.resize((32, 32))  # Resize to 32x32x3.
        data.append(image)  # Append the processed image.

    # Create a DataFrame from the images and labels.
    return pd.DataFrame({'image': data, 'label': labels})

# Image transformation for preprocessing before feeding into a model.
image_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor.
])

# Define a simple CNN model for feature extraction.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Assuming 32x32 input images after pooling.
        self.fc2 = nn.Linear(128, 10)  # Output for 10 classes (e.g., CIFAR-10).

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)  # Flatten the tensor for the fully connected layers.
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to extract features from images using the simple CNN model.
def extract_image_features(images):
    cnn_model = SimpleCNN()
    cnn_model.eval()  # Set the model to evaluation mode.

    features = []  # List to store extracted features.

    for img in images:
        img_tensor = image_transform(img).unsqueeze(0)  # Apply transformation and add batch dimension.
        with torch.no_grad():
            feature = cnn_model(img_tensor)  # Extract features from the image.
        features.append(feature.flatten().numpy())  # Flatten and convert to NumPy array.

    return np.array(features)  # Return as a NumPy array.

# Function for uncertainty sampling based on model predictions.
def uncertainty_sampling(model, X, num_samples):
    probas = model.decision_function(X)  # Get decision function values from the SVM.
    uncertainty = np.max(np.abs(probas), axis=1)  # Calculate uncertainty based on distance from the decision boundary.
    uncertainty_indices = np.argsort(uncertainty)[:num_samples]  # Select indices with the highest uncertainty.
    return uncertainty_indices  # Return the indices.

# Function for entropy sampling based on model predictions.
def entropy_sampling(model, X, num_samples):
    probas = model.predict_proba(X)  # Get predicted probabilities for each class.
    # Calculate entropy for each sample.
    entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)  # Avoid log(0) by adding a small constant.
    entropy_indices = np.argsort(entropy)[-num_samples:]  # Select indices with the highest entropy.
    return entropy_indices  # Return the indices.

# Function to train the model using the training data.
def train_model(X, y):
    # Create a pipeline with an SVM classifier.
    model = SVC(probability=True)  # SVM for image classification.
    model.fit(X, y)  # Train the model on the provided data and labels.
    return model  # Return the trained model.

# Main active learning loop.
def active_learning(json_file):
    # Load the configuration from the JSON file.
    config = load_config(json_file)

    # Read image data from the text files.
    df = read_image_files(config)

    # Extract features from the images using the CNN.
    X = extract_image_features(df['image'])
    y = df['label']  # Labels (annotations).

    # Split the dataset into training and testing sets (80% training, 20% testing).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # Train the initial model using the training data.
    model = train_model(X_train, y_train)

    # Get active learning parameters from the configuration.
    algorithm = config['active_learning_method']
    num_samples = int(len(X_train) * 0.5)  # Number of samples to annotate in each iteration.

    # Active Learning loop for 5 iterations.
    for _ in range(5):
        if len(X_test) == 0:
            print("No more samples left to annotate.")
            break  # Exit if no samples left in the test set.

        current_num_samples = min(num_samples, len(X_test))  # Adjust the number of samples if fewer available.

        # Use the selected algorithm for sampling.
        if algorithm == "uncertainty":
            indices = uncertainty_sampling(model, X_test, current_num_samples)
        elif algorithm == "entropy":
            indices = entropy_sampling(model, X_test, current_num_samples)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Select the samples and their labels directly without reshaping.
        X_selected = X_test[indices]  # No need to reshape.
        y_selected = y_test[indices] # Get corresponding labels.

        # Retrain the model with the newly annotated samples.
        model.fit(np.vstack([X_train, X_selected]), np.hstack([y_train, y_selected]))

        # Remove the selected samples from X_test and y_test.
        X_test = np.delete(X_test, indices, axis=0)
        y_test = np.delete(y_test, indices, axis=0)

    # Save the trained model.
    joblib.dump(model, "trained_model_for_image_annotation")
    print("Model saved successfully.")

# Function to load a trained model from a file.
def load_model(model_path):
    return joblib.load(model_path)

# Function to make predictions and save them in a file.
def predict_and_evaluate(model_path, json_file, output_file="./predicted_image_labels.txt"):
    # Load the trained model.
    model = load_model(model_path)

    # Load the configuration and image data.
    config = load_config(json_file)
    df = read_image_files(config)

    # Extract features from the images.
    X = extract_image_features(df['image'])

    # Predict the labels using the trained model.
    y_pred = model.predict(X)

    # Save predicted labels into a text file.
    with open(output_file, 'w') as f:
        for label in y_pred:
            f.write(f"{label}\n")

    print(f"Predicted labels saved to {output_file}.")
    create_json_config(labels_path=output_file, active_learning_method=config['active_learning_method'])

# Function to create a JSON file with the predicted labels.
def create_json_config(labels_path, active_learning_method, output_json_path="./image_results.json"):
    # Create a dictionary to store the results.
    config_data = {
        "labels": labels_path,
        "active_learning_method": active_learning_method
    }

    # Save the dictionary to a JSON file.
    with open(output_json_path, 'w') as json_file:
        json.dump(config_data, json_file, indent=4)
    
    print(f"JSON configuration saved to {output_json_path}.")