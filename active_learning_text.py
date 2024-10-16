import json  # Import the JSON module to read configuration files.
import numpy as np  # Import NumPy for numerical operations.
import pandas as pd  # Import Pandas for data handling.
from sklearn.feature_extraction.text import CountVectorizer  # For converting text to numerical features.
from sklearn.naive_bayes import MultinomialNB  # Import the Naive Bayes classifier.
from sklearn.pipeline import make_pipeline  # For creating a machine learning pipeline.
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets.
import joblib  # Import joblib for saving the model.
from sklearn.metrics import accuracy_score  # For calculating the accuracy of predictions.

# Function to load JSON configuration from a given file path.
def load_config(file_path):
    with open(file_path, 'r') as f:  # Open the JSON file in read mode.
        return json.load(f)  # Load and return the JSON data as a Python dictionary.

# Function to read text files and create a labeled dataset.
def read_text_files(annotation_config):
    data = []  # Initialize an empty list to store text data.
    labels = []  # Initialize an empty list to store corresponding labels.

    data_path = annotation_config['data']
    with open(data_path, 'r') as f:
            for line in f:  # Read the file line by line.
                    data.append(line.strip().strip('"'))

    labels_path = annotation_config['labels']
    with open(labels_path, 'r') as f:
            for line in f:  # Read the file line by line.
                    labels.append(line.strip().strip('"'))
                    
    # Create a DataFrame from the collected data and labels and return it.
    return pd.DataFrame({'text': data, 'label': labels})

# Active Learning Methods.

# Function for uncertainty sampling based on model predictions.
def uncertainty_sampling(model, X, num_samples):
    probas = model.predict_proba(X)  # Get predicted probabilities for each class.
    uncertainty = 1 - np.max(probas, axis=1)  # Calculate uncertainty as 1 - max probability.
    uncertainty_indices = np.argsort(uncertainty)[-num_samples:]  # Get the indices of the most uncertain samples.
    return uncertainty_indices  # Return the indices.

# Function for entropy sampling based on model predictions.
def entropy_sampling(model, X, num_samples):
    probas = model.predict_proba(X)  # Get predicted probabilities for each class.
    # Calculate entropy for each sample.
    entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)  # Avoid log(0) by adding a small constant.
    entropy_indices = np.argsort(entropy)[-num_samples:]  # Get indices of the samples with highest entropy.
    return entropy_indices  # Return the indices.

# Function to train the model using the training data.
def train_model(X, y):
    # Create a machine learning pipeline with text vectorization and a Naive Bayes classifier.
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X, y)  # Train the model on the provided data and labels.
    return model  # Return the trained model.

# Main active learning loop.
def active_learning(json_file):
    # Load the configuration from the JSON file.
    print("Load the configuration from the JSON file.")
    config = load_config(json_file)
    
    # Read data from the specified text files and create a DataFrame.
    print("Read data from the specified text files and create a DataFrame.")
    df = read_text_files(config)
    
    # Prepare the data for training and testing.
    print("Prepare the data for training and testing.")
    X = df['text']  # Features (text data).
    y = df['label']  # Labels (annotations).

    # Split the initial dataset into training and testing sets (80% training, 20% testing).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the initial model using the training data.
    model = train_model(X_train, y_train)

    # Get active learning parameters from the configuration.
    algorithm = config['active_learning_method']  # Active learning algorithm to use.
    num_samples = int(len(X_train)*0.5) # Number of samples to annotate.

    # Active Learning loop (run for a specified number of iterations).
    for _ in range(5):  # Loop for 5 iterations.
        # Ensure there are enough samples to draw from.
        if len(X_test) == 0:
            print("No more samples left in X_test to annotate.")
            break  # Exit the loop if no samples are left in X_test.
        
        # Check if the number of samples to draw is more than available.
        current_num_samples = min(num_samples, len(X_test))  # Adjust the number of samples to available ones.

        if algorithm == "uncertainty":  # Check if the chosen algorithm is uncertainty sampling.
            indices = uncertainty_sampling(model, X_test, current_num_samples)  # Perform uncertainty sampling.
        elif algorithm == "entropy":  # Check if the chosen algorithm is entropy sampling.
            indices = entropy_sampling(model, X_test, current_num_samples)  # Perform entropy sampling.
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")  # Raise an error for unknown algorithms.

        # Print the selected indices for the chosen algorithm.
        print(f"Selected indices for {algorithm}: {indices}")

        # Retrain the model with the selected samples.
        # Use pd.concat to combine the training data with the newly annotated samples.
        model.fit(pd.concat([X_train, X_test.iloc[indices]]), pd.concat([y_train, y_test.iloc[indices]]))

        # Remove the selected samples from X_test and y_test.
        X_test = X_test.drop(X_test.index[indices])  # Use the indices of X_test to drop the samples.
        y_test = y_test.drop(y_test.index[indices])  # Use the indices of y_test to drop the corresponding labels.

    
    joblib.dump(model, "trained_model_for_text_annotation.joblib")  # Save the trained model to a file.
    print("Model saved successfully.")

# Function to load a trained model from a file.
def load_model(model_path):
    return joblib.load(model_path)  # Load and return the trained model from the specified path.

# Function to load data and labels from a JSON file, predict using a loaded model, and evaluate accuracy.
def predict_and_evaluate(model_path, json_file, output_file="./predicted_labels.txt"):
    # Load the trained model.
    model = load_model(model_path)
    
    # Load the configuration from the JSON file.
    print("Loading data and labels from the JSON file.")
    config = load_config(json_file)
    
    # Read data from the specified text files and create a DataFrame.
    df = read_text_files(config)
    
    # Prepare the data for predictions.
    X = df['text']  # Features (text data).

    # Use the loaded model to predict the labels for the text data.
    y_pred = model.predict(X)

    # Save predicted labels into a text file.
    with open(output_file, 'w') as f:
        for label in y_pred:
            f.write(f"{label}\n")
    
    print(f"Predicted labels saved to {output_file}.")
    create_json_config(labels_path=output_file, active_learning_method=config['active_learning_method'], output_json_path="./results.json")

import json

# Function to create a JSON file with "labels" and "active_learning_method".
def create_json_config(labels_path, active_learning_method, output_json_path="./results.json"):
    # Create a dictionary to store the configuration.
    config_data = {
        "labels": labels_path,  # Path to the predicted labels file.
        "active_learning_method": active_learning_method  # Active learning method used.
    }

    # Write the dictionary to a JSON file.
    with open(output_json_path, 'w') as json_file:
        json.dump(config_data, json_file, indent=4)  # Save with indentation for readability.

    # Notify that the JSON file has been saved.
    print(f"JSON configuration saved to {output_json_path}.")