import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class NeuralNetwork:
    """Neural Network Classifier built from scratch using NumPy"""
    
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        """
        Initialize the neural network
        Args:
            layer_sizes: List of neurons in each layer
            activation: Activation function ('relu', 'sigmoid')
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases"""
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Sigmoid derivative"""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax for multi-class classification"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:  # Output layer
                a = self.softmax(z)
            else:  # Hidden layers
                a = self.relu(z) if self.activation == 'relu' else self.sigmoid(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward_propagation(self, y):
        """Backward pass through the network"""
        m = y.shape[0]
        
        # Output layer error
        delta = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
            
            # Compute delta for previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                if self.activation == 'relu':
                    delta *= self.relu_derivative(self.z_values[i - 1])
                else:
                    delta *= self.sigmoid_derivative(self.activations[i])
    
    def fit(self, X, y, epochs=100, verbose=True):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)
            
            # Backward propagation
            self.backward_propagation(y)
            
            if verbose and (epoch + 1) % 10 == 0:
                loss = -np.mean(y * np.log(output + 1e-8))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward_propagation(X)


if __name__ == "__main__":
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Convert labels to one-hot encoding
    y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train model
    model = NeuralNetwork(
        layer_sizes=[4, 8, 6, 3],  # Input: 4, Hidden: 8, 6, Output: 3
        activation='relu',
        learning_rate=0.1
    )
    
    print("Training Neural Network...")
    model.fit(X_train, y_train, epochs=100, verbose=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Evaluate
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_labels, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test_labels, y_pred)}")
