import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def repeat_experiment(function: callable, X, y, n_exp=200, **kwargs):
    """
    Repeat an experiment multiple times and analyze the results.
    
    Parameters
    ----------
    function : callable
        The optimization function to test (e.g., gradient_descent, adam, etc.)
    X : ndarray
        Design matrix
    y : ndarray
        Target values
    n_exp : int, optional
        Number of experiments to run
    **kwargs : dict
        Additional parameters to pass to the function
        
    Returns
    -------
    dict
        Statistics about the experiments (mean, std, median iterations)
    """
    iterations = []
    for _ in range(n_exp):
        _, iter = function(X, y, **kwargs)
        iterations.append(iter)
    average_iter = np.mean(iterations)
    std_dev_iter = np.std(iterations)
    median_iter = np.median(iterations)
    sns.histplot(iterations, kde=True)
    plt.xlabel("Number of Iterations to Converge")
    plt.ylabel("Frequency")
    cell_text = [
        ["Average Iterations", f"{average_iter:.2f}"],
        ["standard dev", f"{std_dev_iter:.2f}"],
        ["median", f"{median_iter:.2f}"],
        ["analytical", "True" if kwargs.get("analytical", False) else "False"],
    ] + [[k, v] for k, v in kwargs.items()]
    plt.table(
        cellText=cell_text,
        colLabels=["Parameter", "Value"],
        loc="upper left",
        cellLoc="center",
        colWidths=[0.2, 0.2],
        colColours=["lightgrey"] * 2,
    )
    plt.title(f'Distribution of Iterations to Converge (lr={kwargs["lr"]})')
    plt.show()

    return {
        "mean": average_iter,
        "std": std_dev_iter,
        "median": median_iter,
    }