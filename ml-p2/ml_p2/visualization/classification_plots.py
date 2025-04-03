import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def plot_classification_results(model, X, y, title="Classification Results"):
    """
    Plot classification results including:
    1. Decision boundary (if 2D or reduced to 2D)
    2. Training history (accuracy and loss)
    3. Confusion matrix
    """

    fig = plt.figure(figsize=(7, 3))
    
    ax1 = fig.add_subplot(121)
    plot_decision_boundary(ax1, model, X, y)
    ax1.set_title("Decision Boundary")
    
    ax2 = fig.add_subplot(122)
    # plot_training_history(ax2, model)
    plot_decision_boundary(ax2, model, X, y,prob=True)
    ax2.set_title("Probability map")
    
    # ax3 = fig.add_subplot(133)
    # plot_prediction_probabilities(ax3, model, X, y)
    # ax3.set_title("Prediction Probabilities")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(ax, model, X, y,prob:bool=False):
    """Plot decision boundary and scatter plot of data"""
    #Shape mismatch
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

  
    h = 0.02  # Step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Get predictions for mesh grid points
    if X.shape[1] > 2:
        mesh_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    else:
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    if prob:
        Z = model.predict(mesh_points)
    else:
        try:
            Z = model.predict_classes(mesh_points)
        except AttributeError:
            Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.8, s=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax)

def plot_training_history(ax, model):
    """Plot training history (loss and accuracy)"""
    epochs = range(1, len(model.cost_history) + 1)
    ax.plot(epochs, model.cost_history, 'b-', label='Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

def plot_prediction_probabilities(ax, model, X, y):
    """Plot histogram of prediction probabilities"""
    probas = model.predict_proba(X)
    for i in range(2):  # Binary classification
        mask = y.ravel() == i
        ax.hist(probas[mask], bins=20, alpha=0.5, 
                label=f'Class {i}', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.legend()


def plot_decision_boundary2(model, X, y):
    """
    Plot the decision boundary of the model
    
    Parameters:
    -----------
    model : classifier object
        Trained classifier
    X : array-like
        Training data
    y : array-like
        Target values
    """
    # Check if X has more than 2 features
    if X.shape[1] > 2:
        # If more than 2 features, use PCA to reduce to 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    if X.shape[1] > 2:
        mesh_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    else:
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Sklearn Neural Network Decision Boundary')
    plt.show()

def plot_data(X, y, title="Data Distribution"):
    """
    Create a scatter plot of 2D data points colored by their target class
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, 2)
        The input features
    y : array-like of shape (n_samples,)
        The target values
    title : str, default="Data Distribution"
        The title of the plot
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='viridis')
    plt.colorbar(scatter, label='Target Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_accuracy_heatmap(accuracies, param1, param2,param1_label,param2_label,title):
    matrix = np.array(accuracies).reshape(len(param1), len(param2))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=param2,
        yticklabels=param1,
        cmap="viridis",
        annot_kws={'size': 20},
        cbar_kws={'label': 'Accuracy'},
        linewidths=0.5,
        linecolor='black'
    )
    plt.xlabel(param2_label, fontsize=12)
    plt.ylabel(param1_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()


def plot_decision_boundary_multiclass(model, X, y, title="Multi-class Classification"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    mesh_2d = np.c_[xx.ravel(), yy.ravel()]

    feature_means = X[:, 2:].mean(axis=0)
    mesh_points = np.column_stack(
        [mesh_2d, np.tile(feature_means, (mesh_2d.shape[0], 1))]
    )

    Z = model.predict_classes(mesh_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.colorbar(label="Class")
    plt.tight_layout()
    plt.show()
