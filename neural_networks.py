import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        self.activations = {}
        self.gradients = {}
    
    def _activation(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0,Z)
        else:
            raise ValueError (f"Unsupported activation funtion: {self.activation_fn}")

    def _activation_derivative(self, Z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(Z) ** 2
        elif self.activation_fn == 'relu':
            return (Z > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")
        
    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._activation(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        out = Z2
        
        # TODO: store activations for visualization
        self.activations = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'out': out}
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        A1 = self.activations['A1']
        out = self.activations['out']
        
        # TODO: update weights with gradient descent
        m = y.shape[0]
        dZ2 = out - y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims = True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activation_derivative(self.activations['Z1'])
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis = 0, keepdims = True)
        # TODO: store gradients for visualization
        self.gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        pass

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.activations['A1']
    try:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    except Exception as e:
        print(f"Error while plotting hidden features: {e}")
    

    # TODO: Hyperplane visualization in the hidden space
    x_vals = np.linspace(hidden_features[:,0].min(), hidden_features[:,0].max(), 100)
    y_vals = np.linspace(hidden_features[:, 1].min(), hidden_features[:,1].max(), 100)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.c_[xx.ravel(), yy.ravel()]
    hidden_grid = np.dot(grid, mlp.W2) + mlp.b2
    Z = hidden_grid[:,0].reshape(xx.shape)
    ax_hidden.contourf(xx, yy, Z, levels = 50, cmap = 'coolwarm', alpha = 0.3)
   
    # TODO: Distorted input space transformed by the hidden layer
    transformed_input = hidden_features @ mlp.W2 + mlp.b2
    ax_input.scatter(transformed_input[:,0], transformed_input[:,1], c=y.ravel(), cmap = 'bwr', alpha = 0.7)
    ax_input.set_title("Distorted Input Space")
    ax_input.set_xlabel("Input Dim 1")
    ax_input.set_ylabel("Input Dim 2")
   
    # TODO: Plot input layer decision boundary
    x_vals = np.linspace(X[:, 0].min(), X[:,0].max(), 100)
    y_vals = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.c_[xx.ravel(), yy.ravel()]
    input_layer_output = mlp.forward(grid)
    Z = input_layer_output[:,0].reshape(xx.shape)
    ax_input.contour(xx, yy, Z, levels = [0.5], colors = 'k', linestyles = '--', linewidths = 1.5)
    
    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    for i, (W, dW) in enumerate(zip([mlp.W1, mlp.W2], [mlp.gradients['dW1'], mlp.gradients['dW2']])):
        norm = np.linalg.norm(dW, axis=1)
        for j, (w, dw) in enumerate(zip(W, dW)):
            ax_gradient.add_patch(Circle(w, norm[0] * 0.1, color='r', alpha=0.3))
            ax_gradient.arrow(w[0], w[1], -dw[0], -dw[1], head_width=0.05, head_length=0.1, fc='blue', ec='blue', alpha=0.6)

    ax_gradient.set_title("Gradient Visualization")
    ax_gradient.set_xlim([-1, 1])
    ax_gradient.set_ylim([-1, 1])

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)