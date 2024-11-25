import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial

# Directory for saving results
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_name = activation

        # Initialize activations and their derivatives
        if activation == 'tanh':
            self.activation_fn = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'sigmoid':
            self.activation_fn = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip for stability
            self.activation_derivative = lambda x: x * (1 - x)
        elif activation == 'relu':
            self.activation_fn = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Initialize weights with He initialization for ReLU
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.bias_output = np.zeros((1, output_dim))

        self.gradients = {
            "weights_input_hidden": np.zeros_like(self.weights_input_hidden),
            "weights_hidden_output": np.zeros_like(self.weights_hidden_output),
        }


    def forward(self, X):
        self.hidden_pre_activation = X @ self.weights_input_hidden + self.bias_hidden
        self.hidden_pre_activation = np.clip(self.hidden_pre_activation, -1e6, 1e6)  # Prevent overflow
        self.hidden_activations = self.activation_fn(self.hidden_pre_activation)

        self.output_pre_activation = self.hidden_activations @ self.weights_hidden_output + self.bias_output
        self.output_pre_activation = np.clip(self.output_pre_activation, -1e6, 1e6)  # Prevent overflow
        self.output = self.activation_fn(self.output_pre_activation)

        self.output = np.clip(self.output, 1e-15, 1 - 1e-15)  # Avoid log(0)
        return self.output

    def backward(self, X, y):
        y = (y + 1) / 2  # Normalize labels
        output_delta = self.output - y

        hidden_error = output_delta @ self.weights_hidden_output.T
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_activations)

        self.gradients["weights_input_hidden"] = np.abs(X.T @ hidden_delta)
        self.gradients["weights_hidden_output"] = np.abs(self.hidden_activations.T @ output_delta)

        # Gradient clipping to prevent exploding gradients
        np.clip(output_delta, -1e2, 1e2, out=output_delta)
        np.clip(hidden_delta, -1e2, 1e2, out=hidden_delta)

        # Update weights
        self.weights_hidden_output -= self.lr * self.hidden_activations.T @ output_delta
        self.bias_output -= self.lr * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden -= self.lr * X.T @ hidden_delta
        self.bias_hidden -= self.lr * np.sum(hidden_delta, axis=0, keepdims=True)


def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    ######################### LEFT ###################
    # Get hidden features
    hidden_features = mlp.hidden_activations

    # Plot hidden space in 3D
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )

    # Define the 3D grid in the hidden feature space
    x_min, x_max = hidden_features[:, 0].min() - 1, hidden_features[:, 0].max() + 1
    y_min, y_max = hidden_features[:, 1].min() - 1, hidden_features[:, 1].max() + 1
    z_min, z_max = hidden_features[:, 2].min() - 1, hidden_features[:, 2].max() + 1

    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, 30),
        np.linspace(y_min, y_max, 30),
        np.linspace(z_min, z_max, 30)
    )

    # Compute the non-linear transformation by the hidden layer (curved blue boundary)
    hidden_transformation = (
        mlp.weights_input_hidden[0, 0] * xx +
        mlp.weights_input_hidden[1, 0] * yy
    )

    if mlp.weights_input_hidden.shape[0] > 2:
        hidden_transformation += mlp.weights_input_hidden[2, 0] * zz

    # Apply the dynamically chosen activation function (tanh, sigmoid, relu)
    hidden_transformation = mlp.activation_fn(hidden_transformation)

    # Plot the curved blue boundary
    ax_hidden.plot_surface(
        xx[:, :, 0], yy[:, :, 0], hidden_transformation[:, :, 0],
        alpha=0.5, color="blue"
    )

    # Compute and plot the tan hyperplane in the hidden space
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    zz = -(mlp.weights_hidden_output[0, 0] * xx +
        mlp.weights_hidden_output[1, 0] * yy +
        mlp.bias_output[0, 0]) / (mlp.weights_hidden_output[2, 0] + 1e-5)

    ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='tan')

    # Set plot limits and title
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    ax_hidden.set_xlim(x_min, x_max)
    ax_hidden.set_ylim(y_min, y_max)
    ax_hidden.set_zlim(hidden_features[:, 2].min() - 1, hidden_features[:, 2].max() + 1)

    ##################### MIDDLE #####################
    
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title("Input Space")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, levels=50, cmap="bwr", alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor='k', cmap='bwr')
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    ##################### RIGHT ######################

    # Node positions
    input_nodes = [0.2, 0.8]  # Two input nodes on the left
    hidden_nodes = [0.2, 0.5, 0.8]  # Three hidden nodes in the middle
    output_node = [0.5]  # Single output node on the far right

    # Layer heights
    input_layer_x = 0.0
    hidden_layer_x = 0.5
    output_layer_x = 1.0

    # Plot input-to-hidden connections
    for i, y1 in enumerate(input_nodes):
        for j, y2 in enumerate(hidden_nodes):
            # Adjust thickness based on gradient magnitude
            thickness = mlp.gradients["weights_input_hidden"][i, j] * 2
            ax_gradient.plot(
                [input_layer_x, hidden_layer_x],  # Horizontal connection
                [y1, y2],  # Vertical positions
                'purple',
                linewidth=max(thickness, 0.5)  # Minimum thickness for visibility
            )

    # Plot hidden-to-output connections
    for i, y1 in enumerate(hidden_nodes):
        for j, y2 in enumerate(output_node):
            thickness = mlp.gradients["weights_hidden_output"][i, j] * 2
            ax_gradient.plot(
                [hidden_layer_x, output_layer_x],  # Horizontal connection
                [y1, y2],  # Vertical positions
                'purple',
                linewidth=max(thickness, 0.5)
            )

    # Plot nodes
    ax_gradient.scatter([input_layer_x] * len(input_nodes), input_nodes,
                        c='blue', s=300, label="Input Nodes")
    ax_gradient.scatter([hidden_layer_x] * len(hidden_nodes), hidden_nodes,
                        c='blue', s=300, label="Hidden Nodes")
    ax_gradient.scatter([output_layer_x] * len(output_node), output_node,
                        c='blue', s=300, label="Output Node")

    # Annotate nodes
    ax_gradient.text(input_layer_x + 0.06, input_nodes[0] + 0.01, 'x1', ha='center', fontsize=12)
    ax_gradient.text(input_layer_x + 0.06, input_nodes[1] + 0.01, 'x2', ha='center', fontsize=12)
    ax_gradient.text(hidden_layer_x + 0.06, hidden_nodes[0] + 0.01, 'h1', ha='center', fontsize=12)
    ax_gradient.text(hidden_layer_x + 0.06, hidden_nodes[1] + 0.01, 'h2', ha='center', fontsize=12)
    ax_gradient.text(hidden_layer_x + 0.06, hidden_nodes[2] + 0.01, 'h3', ha='center', fontsize=12)
    ax_gradient.text(output_layer_x + 0.06, output_node[0] + 0.01, 'y', ha='center', fontsize=12)

    # Title and limits
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")

    ###############################################

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "relu"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)