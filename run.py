import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def load_dataset():
    # Define the directory containing the CSV files
    directory = "raw dataset"

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each CSV file
    for file in csv_files:
        file_path = os.path.join(directory, file)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add "signal" column: 1 if the file is "signalHH.csv", otherwise 0
        df["signal"] = 1 if file == "signalHH.csv" else 0
        
        # Append to the list
        dfs.append(df)

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df

def MDS_visualization(df_whole):
    df = df_whole.sample(n=500, random_state=40)

    # Separate features and target variable
    X = df.drop(columns=["signal"])
    y = df["signal"]

    # Standardize the data (MDS works better with scaled data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute total variance in the original high-dimensional space
    original_variance = np.sum(np.var(X_scaled, axis=0))

    # Perform MDS (reducing to 2 dimensions for visualization)
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
    X_mds = mds.fit_transform(X_scaled)

    # Compute variance explained in 2D space
    mds_variance = np.sum(np.var(X_mds, axis=0))
    variance_explained_ratio = mds_variance / original_variance  # Ratio of retained variance

    # Convert to DataFrame
    mds_df = pd.DataFrame(X_mds, columns=["MDS1", "MDS2"])
    mds_df["signal"] = y.values  # Add the signal column

    # Plot MDS results
    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}  # Define colors for each class
    labels = {0: 'Non-Signal', 1: 'Signal'}

    for label in [0, 1]:
        subset = mds_df[mds_df["signal"] == label]
        plt.scatter(subset["MDS1"], subset["MDS2"], c=colors[label], label=labels[label], alpha=0.6)

    # Add variance explained in title
    plt.xlabel("MDS Component 1")
    plt.ylabel("MDS Component 2")
    plt.title(f"MDS Plot of Dataset with 1000 random samples (Explained Variance: {variance_explained_ratio:.2%})")

    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG image
    plt.savefig("MDS_plot.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

def logistic_regression(X_train, X_test, y_train, y_test):
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report for Logistic Regression:\n", classification_report(y_true=y_test, y_pred=y_pred))

def svm_classifier(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1.0)  # Radial Basis Function (RBF) kernel
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.4f}")
    print("SVM Classification Report for SVM:\n", classification_report(y_test, y_pred))

def mlp_classifier(X_train, X_test, y_train, y_test):
    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define MLP model
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                          max_iter=500, random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"MLP Accuracy: {accuracy:.4f}")
    print("MLP Classification Report with MLP NN:\n", classification_report(y_test, y_pred))

def reduce_dataset(X_train, X_test, y_train, y_test, train_size=0.2, test_size=0.2):
    """ Downsamples the dataset while preserving class balance. """
    X_train_reduced, _, y_train_reduced, _ = train_test_split(
        X_train, y_train, train_size=train_size, stratify=y_train, random_state=42
    )
    
    X_test_reduced, _, y_test_reduced, _ = train_test_split(
        X_test, y_test, train_size=test_size, stratify=y_test, random_state=42
    )
    
    return X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced

if __name__ == "__main__":

    df = load_dataset()
    print(df.head())

    # Shuffle dataset to ensure randomness
    df = df.sample(frac=1, random_state=41).reset_index(drop=True)

    # MDS_visualization(df)

    # Separate features and target variable
    X = df.drop(columns=["signal"])  # Features (excluding the target)
    y = df["signal"]  # Target variable (0 or 1)

    # Split dataset into train (80%) and test (20%) ensuring stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=41)

    logistic_regression(X_train, X_test, y_train, y_test)

    # svm_classifier(X_train, X_test, y_train, y_test)

    # Reduce training dataset size
    # Call function to reduce dataset
    X_train_small, X_test_small, y_train_small, y_test_small = reduce_dataset(X_train, X_test, y_train, y_test, train_size=0.2, test_size=0.2)

    mlp_classifier(X_train_small, X_test_small, y_train_small, y_test_small)