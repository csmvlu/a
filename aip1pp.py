print("BFS")
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Define the graph
graph = {
    "Mahavir Nagar": ["satya Nagar Rd", "Boraspada Road"],
    "satya Nagar Rd": ["RM Bhattad Rd"],
    "RM Bhattad Rd": ["Dattapada Road"],
    "Dattapada Road": ["Western Express Hwy"],
    "Western Express Hwy": ["jijamata Rd"],
    "jijamata Rd": ["Nicolas Wadi Rd"],
    "Nicolas Wadi Rd": ["Old Nagardas Road"],
    "Old Nagardas Road": ["MVLU"],
    "MVLU": [],
    "Boraspada Road": ["New Link Road"],
    "New Link Road": ["Malad Marve Road"],
    "Malad Marve Road": ["Father Justin Dsouza Rd"],
    "Father Justin Dsouza Rd": ["Datta Mandir Road"],
    "Datta Mandir Road": ["Western Express Hwy"]
}

# BFS function to find the shortest path from source to goal
def breadth_first_search(graph, source, goal):
    queue = deque([[source]])  # Initialize the queue with the source path
    visited = {source}  # Track visited nodes
    while queue:
        path = queue.popleft()  # Get the first path from the queue
        current = path[-1]  # Get the last node in the current path
        if current == goal:
            return path  # Return the path if the goal is reached
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)  # Mark the neighbor as visited
                new_path = path + [neighbor]  # Create a new path
                queue.append(new_path)  # Add the new path to the queue
            else:
                print("visited:", new_path)  
    return None  # Return None if no path is found

# Run BFS to get the shortest path from "Mahavir Nagar" to "MVLU"
bfs_path = breadth_first_search(graph, "Mahavir Nagar", "MVLU")

# Create and draw the graph
G = nx.DiGraph(graph)  # Create a directed graph
pos = nx.spring_layout(G, seed=75)  # Position the nodes using a spring layout
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue',
        font_size=10, font_weight='bold', edge_color='gray')

# Highlight the BFS path on the graph
if bfs_path:
    path_edges = list(zip(bfs_path, bfs_path[1:]))  # Create a list of edges in the path
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)  # Highlight the path edges in red
    nx.draw_networkx_nodes(G, pos, nodelist=bfs_path, node_color='red', node_size=500)  # Highlight the path nodes in red

# Create a legend for the graph
plt.legend(handles=[
    plt.Line2D([0], [0], color='gray', label='Normal Path'),
    plt.Line2D([0], [0], color='red', label='BFS Path')
], loc='upper right')

# Show the plot
plt.show()

print("A* Algorithm---------------------------------------------------------------------------------------------------------------------------------")
import networkx as nx
import matplotlib.pyplot as plt

# Define the graph nodes with weights
Graph_nodes = {
    "Sai Kripa": [("Hanjer Road", 2)],
    "Hanjer Road": [("Jijamata Road", 2)],
    "Jijamata Road": [("Sardar Dairy", 3)],
    "Sardar Dairy": [("Highway", 2)],
    "Highway": [("Koila Compound", 3), ("Vraj Group", 2)],
    "Koila Compound": [("Bhuta School", 1)],
    "Bhuta School": [("Nagardas", 2)],
    "Nagardas": [("MVLU", 1)],
    "MVLU": [],
    "Vraj Group": [("Parsi", 1)],
    "Parsi": [("Nancy", 2)],
    "Nancy": [("Nagardas", 1)]
}

# Function to get neighbors with weights
def get_neighbors(v):
    """Return the neighbors of the given node with weights."""
    return Graph_nodes.get(v, [])

# Heuristic function
def h(n):
    """Return the heuristic distance to the goal for the given node."""
    H_dist = {
        'Sai Kripa': 10,
        'Hanjer Road': 9,
        'Jijamata Road': 8,
        'Sardar Dairy': 7,
        'Highway': 6,
        'Koila Compound': 5,
        'Vraj Group': 4,
        'Bhuta School': 4,
        'Nagardas': 2,
        'MVLU': 0,
        'Parsi': 5,
        'Nancy': 3
    }
    return H_dist.get(n, float('inf'))  # Return infinity if node not found

# A* algorithm implementation
def aStarAlgo(start_node, stop_node):
    """Perform the A* search algorithm."""
    open_set = {start_node}
    closed_set = set()
    g = {start_node: 0}
    parents = {start_node: start_node}

    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n is None or g[v] + h(v) < g[n] + h(n):
                n = v
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            # Create a list of edges in the path
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            print('Path found: {}'.format(path))
            print('Path edges: {}'.format(path_edges))  # Print the edges in the path
            return path

        open_set.remove(n)
        closed_set.add(n)
        for (m, weight) in get_neighbors(n):
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight  # Use the actual weight of the edge
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

    print('Path does not exist!')
    return None

# Run A* Search
path = aStarAlgo('Sai Kripa', 'MVLU')

# Create a directed graph
G = nx.DiGraph()
# Add nodes to the graph
for node in Graph_nodes:
    G.add_node(node)

# Add edges to the graph with weights
for node, neighbors in Graph_nodes.items():
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

# Use spring layout for better positioning
pos = nx.spring_layout(G, seed=6)  # Adjust seed for better spacing

# Draw the graph without node labels
plt.figure(figsize=(12, 12))  # Increase figure size for clarity
nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', font_size=10)

# Highlight the path in red
if path:
    # Highlight nodes in the path
    red_nodes = nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', node_size=500)
    # Highlight edges in the path
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    red_edges = nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)

# Draw edge weights with smaller font size
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)  # Smaller font size

# Add heuristic values as labels
heuristic_labels = {node: f"{node}: {h(node)}" for node in Graph_nodes}
nx.draw_networkx_labels(G, pos, labels=heuristic_labels, font_color='black', verticalalignment='bottom', font_size=8)  # Smaller font size

# Set title and display the graph
plt.title('Graph Representation with A* Path Highlighted')
plt.axis('off')
plt.show()

print("Decision Tree---------------------------------------------------------------------------------------------------------------------------------#)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree  # Import the tree module for visualizing the decision tree
print("Libraries imported")
# Load the Iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
print("Dataframe of dataset created")

# Rename columns for easier access
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Check for any missing values
print("Null values in each column:\n", df.isnull().sum())

# Encode the target labels as integers (setosa=0, versicolor=1, virginica=2)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# View the first few rows of the dataframe
print(df.head())

# X - Features, y - Label
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0, criterion='gini')
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions_test = clf.predict(X_test)
print("Accuracy on test set: ", accuracy_score(y_test, predictions_test) * 100)

# Make predictions on the training set
predictions_train = clf.predict(X_train)
print("Accuracy on training set: ", accuracy_score(y_train, predictions_train) * 100)

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=['sepal_length', 'sepal_width', 'petal_length',
                                                 'petal_width'], class_names=iris.target_names)
plt.show()

print("2nd")

# Additional plot of the Decision Tree without feature and class names for clarity
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True)
plt.show()

# Print classification report and confusion matrix for test set
print("Classification Report (Test Set):\n", classification_report(y_test, predictions_test))
print("Confusion Matrix (Test Set):\n", confusion_matrix(y_test, predictions_test))

# Print classification report and confusion matrix for training set
print("Classification Report (Training Set):\n", classification_report(y_train, predictions_train))
print("Confusion Matrix (Training Set):\n", confusion_matrix(y_train, predictions_train))

print("Feed Forward Back Propagation---------------------------------------------------------------------------------------------------------------------------------#)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load and Prepare the Data
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for visualization
data = pd.DataFrame(X, columns=iris.feature_names)
data['target'] = y
print("T083 Pratham ")
print(data.describe())
print()
print("Shape of the dataset is:", data.shape)
print()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Build and Train the Feed Forward Neural Network
# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network parameters
input_layer_size = X_train.shape[1]
hidden_layer_size = 10  # Number of neurons in hidden layer
output_layer_size = 3    # Three classes in the Iris dataset
learning_rate = 0.01
epochs = 10000

# Initialize weights
np.random.seed(42)
weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)

# Training the Neural Network
mse_values = []
for epoch in range(epochs):
    # Feedforward
    hidden_layer_input = np.dot(X_train, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_output = sigmoid(final_layer_input)
    
    # One-hot encode the output
    y_train_onehot = np.zeros((y_train.size, y_train.max() + 1))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    
    # Compute the error
    error = y_train_onehot - final_output
    
    # Backpropagation
    d_final_output = error * sigmoid_derivative(final_output)
    error_hidden_layer = np.dot(d_final_output, weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights
    weights_hidden_output += np.dot(hidden_layer_output.T, d_final_output) * learning_rate
    weights_input_hidden += np.dot(X_train.T, d_hidden_layer) * learning_rate
    
    mse = np.mean(error**2)
    mse_values.append(mse)
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs}, MSE: {mse}')

# Visualize the training process
plt.figure(figsize=(10, 6))
plt.plot(mse_values, label="MSE during Training")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Progress of the Neural Network")
plt.legend()
plt.show()

# Step 3: Evaluate the Neural Network
hidden_layer_input_test = np.dot(X_test, weights_input_hidden)
hidden_layer_output_test = sigmoid(hidden_layer_input_test)
final_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output)
final_output_test = sigmoid(final_layer_input_test)

# Convert final output to class predictions
predictions = np.argmax(final_output_test, axis=1)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
conf_matrix = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of the Neural Network Predictions")
plt.show()
print("SVM------------------------------------------------------------------------------------------------------------------------------#)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Function to visualize confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=100)
    plt.title(title)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Classes')
    plt.show()

# Step 1: Classification before adding noise
print("Classification before adding noise:")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.3, random_state=42)

# Create and train the SVM model (Linear)
svm_model_linear = SVC(kernel='linear', C=1)
svm_model_linear.fit(X_train, y_train)

# Make predictions
y_pred_linear = svm_model_linear.predict(X_test)

# Evaluate the model
print("Accuracy before adding noise (Linear):", metrics.accuracy_score(y_test, y_pred_linear))
print("F1 Score before adding noise (Linear):", metrics.f1_score(y_test, y_pred_linear, average='macro'))
print("Precision before adding noise (Linear):", metrics.precision_score(y_test, y_pred_linear, average='macro'))
print("Recall before adding noise (Linear):", metrics.recall_score(y_test, y_pred_linear, average='macro'))

# Confusion Matrix for Linear model
conf_matrix_linear = metrics.confusion_matrix(y_test, y_pred_linear)
print("\nConfusion Matrix (Before Noise - Linear):")
print(conf_matrix_linear)

# Plot decision boundaries for Linear model
plot_decision_boundaries(X_train, y_train, svm_model_linear, title='Decision Boundaries (Linear Model)')

# Step 2: Classification after adding noise
print("\nClassification after adding noise:")

# Add random noise to the data
np.random.seed(42)
noise = np.random.normal(0, 0.5, X.shape)
X_noisy = X + noise

# Split the noisy data into training and testing sets
X_train_noisy, X_test_noisy, y_train, y_test = train_test_split(X_noisy[:, :2], y, test_size=0.3, random_state=42)

# Create and train the SVM model on noisy data (Linear)
svm_model_noisy_linear = SVC(kernel='linear', C=1)
svm_model_noisy_linear.fit(X_train_noisy, y_train)

# Make predictions on the noisy test set (Linear)
y_pred_noisy_linear = svm_model_noisy_linear.predict(X_test_noisy)

# Evaluate the model
print("Accuracy after adding noise (Linear):", metrics.accuracy_score(y_test, y_pred_noisy_linear))
print("F1 Score after adding noise (Linear):", metrics.f1_score(y_test, y_pred_noisy_linear, average='macro'))
print("Precision after adding noise (Linear):", metrics.precision_score(y_test, y_pred_noisy_linear, average='macro'))
print("Recall after adding noise (Linear):", metrics.recall_score(y_test, y_pred_noisy_linear, average='macro'))

# Confusion Matrix for Noisy Linear model
conf_matrix_noisy_linear = metrics.confusion_matrix(y_test, y_pred_noisy_linear)
print("\nConfusion Matrix (After Noise - Linear):")
print(conf_matrix_noisy_linear)

# Plot decision boundaries for Noisy Linear model
plot_decision_boundaries(X_train_noisy, y_train, svm_model_noisy_linear, title='Decision Boundaries (Noisy Linear Model)')

# Create and train the SVM model on noisy data (Non-Linear)
svm_model_noisy_non_linear = SVC(kernel='rbf', C=1)
svm_model_noisy_non_linear.fit(X_train_noisy, y_train)

# Make predictions on the noisy test set (Non-Linear)
y_pred_noisy_non_linear = svm_model_noisy_non_linear.predict(X_test_noisy)

# Evaluate the non-linear model
print("Accuracy after adding noise (Non-Linear):", metrics.accuracy_score(y_test, y_pred_noisy_non_linear))
print("F1 Score after adding noise (Non-Linear):", metrics.f1_score(y_test, y_pred_noisy_non_linear, average='macro'))
print("Precision after adding noise (Non-Linear):", metrics.precision_score(y_test, y_pred_noisy_non_linear, average='macro'))
print("Recall after adding noise (Non-Linear):", metrics.recall_score(y_test, y_pred_noisy_non_linear, average='macro'))

# Confusion Matrix for Noisy Non-Linear model
conf_matrix_noisy_non_linear = metrics.confusion_matrix(y_test, y_pred_noisy_non_linear)
print("\nConfusion Matrix (After Noise - Non-Linear):")
print(conf_matrix_noisy_non_linear)

# Visualize confusion matrices
plot_confusion_matrix(conf_matrix_linear, title='Confusion Matrix (Before Noise - Linear)')
plot_confusion_matrix(conf_matrix_noisy_linear, title='Confusion Matrix (After Noise - Linear)')
plot_confusion_matrix(conf_matrix_noisy_non_linear, title='Confusion Matrix (After Noise - Non-Linear)')

print("AdaboostEnsemble ---------------------------------------------------------------------------------------------------------------------------------#)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for creating the accuracy comparison table
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train AdaBoost classifier with default base estimator (Decision Tree)
abc_default = AdaBoostClassifier(n_estimators=50, learning_rate=1, algorithm='SAMME')  # Use 'SAMME' to avoid warning
model_default = abc_default.fit(X_train, y_train)

# Predict the response for the test dataset using the default model
y_pred_default = model_default.predict(X_test)

# Calculate and print the accuracy of the default model
accuracy_default = metrics.accuracy_score(y_test, y_pred_default)
print("Default AdaBoost Accuracy:", accuracy_default)

# Create and train AdaBoost classifier with Support Vector Classifier as base estimator
svc = SVC(probability=True, kernel='linear')
abc_svc = AdaBoostClassifier(n_estimators=50, estimator=svc, learning_rate=1, algorithm='SAMME')  # Use 'SAMME' to avoid warning
model_svc = abc_svc.fit(X_train, y_train)

# Predict the response for the test dataset using the SVC model
y_pred_svc = model_svc.predict(X_test)

# Calculate and print the accuracy of the SVC model
accuracy_svc = metrics.accuracy_score(y_test, y_pred_svc)
print("SVC AdaBoost Accuracy:", accuracy_svc)

# Create a Voting Classifier combining both models with soft voting
voting_clf = VotingClassifier(estimators=[
    ('default_ada', model_default),
    ('svc_ada', model_svc)
], voting='soft')  # Use 'soft' for averaging predicted probabilities

# Train Voting Classifier
voting_clf.fit(X_train, y_train)

# Predict using Voting Classifier
y_pred_voting = voting_clf.predict(X_test)

# Calculate and print the accuracy of the Voting Classifier
accuracy_voting = metrics.accuracy_score(y_test, y_pred_voting)
print("Voting Classifier Accuracy:", accuracy_voting)

# Evaluation Metrics for Default Model
print("\nClassification Report for Default Model:")
print(classification_report(y_test, y_pred_default))

# Evaluation Metrics for SVC Model
print("\nClassification Report for SVC Model:")
print(classification_report(y_test, y_pred_svc))

# Evaluation Metrics for Voting Classifier
print("\nClassification Report for Voting Classifier:")
print(classification_report(y_test, y_pred_voting))

# Confusion Matrices
confusion_default = confusion_matrix(y_test, y_pred_default)
confusion_svc = confusion_matrix(y_test, y_pred_svc)
confusion_voting = confusion_matrix(y_test, y_pred_voting)

# Plotting Confusion Matrices
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
metrics.ConfusionMatrixDisplay(confusion_default).plot(ax=ax[0], cmap='Blues')
ax[0].set_title('Confusion Matrix - Default AdaBoost')
metrics.ConfusionMatrixDisplay(confusion_svc).plot(ax=ax[1], cmap='Blues')
ax[1].set_title('Confusion Matrix - SVC AdaBoost')
metrics.ConfusionMatrixDisplay(confusion_voting).plot(ax=ax[2], cmap='Blues')
ax[2].set_title('Confusion Matrix - Voting Classifier')
plt.show()

# Adding space between confusion matrices and histogram plot
plt.subplots_adjust(top=0.85)  # Adjust top spacing if needed

# Accuracy Comparison Histogram (Bar Chart)
plt.figure(figsize=(8, 6))
models = ['Default AdaBoost', 'SVC AdaBoost']
accuracies = [accuracy_default * 100, accuracy_svc * 100]
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.title('Accuracy Comparison: AdaBoost vs SVC')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y')

# Adding data labels on top of bars
for index, value in enumerate(accuracies):
    plt.text(index, value + 1, f'{value:.2f}%', ha='center')

plt.show()

print("Naive bayes---------------------------------------------------------------------------------------------------------------------------------#)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Loading the dataset from a URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names=col_names)

# Selecting features and target variable
X = dataset.iloc[:, [0, 3]].values  # Features: Sepal Length and Petal Width
y = dataset.iloc[:, -1].values  # Target: Species

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling using MinMaxScaler to avoid negative values
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Gaussian Naive Bayes model on the Training set
gaussian_classifier = GaussianNB()
gaussian_classifier.fit(X_train, y_train)

# Predicting the Test set results with Gaussian Naive Bayes
y_pred_gaussian = gaussian_classifier.predict(X_test)

# Training the Multinomial Naive Bayes model on the Training set
multinomial_classifier = MultinomialNB()
multinomial_classifier.fit(X_train, y_train)

# Predicting the Test set results with Multinomial Naive Bayes
y_pred_multinomial = multinomial_classifier.predict(X_test)

# Evaluating Gaussian Naive Bayes model
print("Gaussian Naive Bayes Predictions: ", y_pred_gaussian)
print("~" * 20)
ac_gaussian = accuracy_score(y_test, y_pred_gaussian)
print("Gaussian Model Accuracy: ", ac_gaussian * 100, "%")
print("~" * 20)
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian)
print("Gaussian Model Confusion Matrix: ")
print(cm_gaussian)
print("Classification Report for Gaussian Naive Bayes:")
print(classification_report(y_test, y_pred_gaussian, zero_division=0))  # Add zero_division here

# Evaluating Multinomial Naive Bayes model
print("Multinomial Naive Bayes Predictions: ", y_pred_multinomial)
print("~" * 20)
ac_multinomial = accuracy_score(y_test, y_pred_multinomial)
print("Multinomial Model Accuracy: ", ac_multinomial * 100, "%")
print("~" * 20)
cm_multinomial = confusion_matrix(y_test, y_pred_multinomial)
print("Multinomial Model Confusion Matrix: ")
print(cm_multinomial)
print("Classification Report for Multinomial Naive Bayes:")
print(classification_report(y_test, y_pred_multinomial, zero_division=0))  # Add zero_division here

# Visualization of Confusion Matrices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm_gaussian, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_multinomial, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title('Multinomial Naive Bayes Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# Visualizing Accuracy Over Time (for both models)
plt.figure(figsize=(8, 4))
models = ['Gaussian Naive Bayes', 'Multinomial Naive Bayes']
accuracies = [ac_gaussian * 100, ac_multinomial * 100]
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.ylim(0, 100)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.show()
print("KNN---------------------------------------------------------------------------------------------------------------------------------#)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load Dataset (using Iris dataset as an example)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['TARGET CLASS'] = iris.target

# Data Preprocessing (Scaling)
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30)

# List of K values to evaluate
k_values = [1, 5, 10, 23]

# Loop through different K values and evaluate
for k in k_values:
    print(f"\n### Evaluating for K = {k} ###")
    
    # KNN Model with current K
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Predictions
    pred = knn.predict(X_test)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, pred)
    
    # Visualizing the Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g')
    plt.title(f'Confusion Matrix for K={k}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Classification Report
    print(f'Classification Report for K={k}:')
    print(classification_report(y_test, pred))

# Choosing the Best K Value using Error Rate (Elbow Method)
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plotting Error Rate vs K Value
plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

print("Association Mining---------------------------------------------------------------------------------------------------------------------------------#)

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Define the URL to a small sample dataset (mock example)
# Replace this with a real transactional dataset for real-world use
data = pd.DataFrame({
    'Transaction': ['T1', 'T1', 'T2', 'T2', 'T3', 'T3'],
    'Item': ['Milk', 'Bread', 'Milk', 'Diaper', 'Bread', 'Eggs']
})

# Preprocess data into a one-hot encoded DataFrame
basket = (data.groupby(['Transaction', 'Item'])['Item']
          .count().unstack(fill_value=0).reset_index() # Use fill_value=0 to avoid NaN
          .set_index('Transaction'))
basket = basket.astype(bool).astype(int)  # Convert to boolean and then to int

# Generate frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Plotting support vs confidence
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules, x='support', y='confidence', size='lift', sizes=(20, 200), alpha=0.5)
plt.title('Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid()
plt.show()

# Visualizing Evaluation Metrics: Support, Confidence, and Lift
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Convert frozenset to string for plotting purposes
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))

# Support Bar Plot
sns.barplot(ax=axes[0], x='antecedents_str', y='support', data=rules)
axes[0].set_title('Support of Rules')
axes[0].set_ylabel('Support')

# Confidence Bar Plot
sns.barplot(ax=axes[1], x='antecedents_str', y='confidence', data=rules)
axes[1].set_title('Confidence of Rules')
axes[1].set_ylabel('Confidence')

# Lift Bar Plot
sns.barplot(ax=axes[2], x='antecedents_str', y='lift', data=rules)
axes[2].set_title('Lift of Rules')
axes[2].set_ylabel('Lift')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display rules in a well-formatted manner
print("\nAssociation Rules:")
for index, row in rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    print(f"Rule: If you buy {antecedents}, you are likely to also buy {consequents}.")
    print(f"Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}\n")
