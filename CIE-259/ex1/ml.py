import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import (
    Tuple,
    Dict,
    Any,
    Callable,
    List,
)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.preprocessing import (
    PolynomialFeatures,
)
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

N_POINTS = 150  # Total number of data points to generate for each dataset
X_MIN, X_MAX = -10, 5  # The range [min, max) for the input feature X
NOISE_STD_LIN = (
    1.0  # Standard deviation of the Gaussian noise added to the linear data's y values
)

NOISE_STD_POLY = 1.0  # Standard deviation of the noise for the polynomial data
NOISE_STD_TREE = 0.3  # Standard deviation of the noise for the tree-like data
TEST_SIZE = 0.2  # Proportion of the dataset to use as the test set (20%)
RANDOM_STATE = (
    42  # Seed for random number generators to ensure reproducibility of results
)
N_PLOT_POINTS = 500  # Number of points to use for plotting the smooth model lines
POLYNOMIAL_DEGREES_TO_TRY = list(
    range(1, 6)
)  # List of polynomial degrees (1 to 5) to test during tuning
TREE_DEPTHS_TO_TRY = list(
    range(1, 1000)
)  # List of decision tree maximum depths (1 to 9) to test
CV_FOLDS = 5  # Number of folds for cross-validation during hyperparameter tuning

# Set the global random seed for NumPy.
np.random.seed(RANDOM_STATE)


# --- Data Generation Functions ---
def generate_linear_data(
    n_points: int = N_POINTS,
    x_min: float = X_MIN,
    x_max: float = X_MAX,
    noise_std: float = NOISE_STD_LIN,
) -> Tuple[
    np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], Dict[str, float]
]:
    """
    Generates data following a linear trend (y = mx + c) with added random noise.

    Args:
        n_points: Number of data points to generate.
        x_min: Minimum value for the feature X.
        x_max: Maximum value for the feature X.
        noise_std: Standard deviation of the Gaussian noise added to y.

    Returns:
        A tuple containing:
            - X: The input feature array (shape: [n_points, 1]).
            - y: The target variable array (shape: [n_points, 1]).
            - true_func: A function representing the true underlying linear relationship (without noise).
            - true_params: A dictionary containing the true slope and intercept.
    """

    # Generate random X values uniformly distributed between x_min and x_max.
    # .rand(n_points, 1) creates an array of shape (n_points, 1) with values in [0, 1).
    # Scaling and shifting maps these values to the desired [x_min, x_max) range.
    X = (x_max - x_min) * np.random.rand(n_points, 1) + x_min

    # Define the true parameters of the underlying linear relationship
    true_slope = np.random.uniform(1, 4)  # Random slope between 1 and 4
    true_intercept = np.random.uniform(-5, 5)  # Random intercept between -5 and 5

    # True linear function
    true_func = lambda x: true_intercept + true_slope * x

    # Calculate the y values based on the true function
    y_true = true_func(X)

    # Add Gaussian (normal) noise to the true y values to simulate real-world data
    # np.random.randn generates values from a standard normal distribution (mean 0, std 1).
    # Multiplying by noise_std scales the noise.
    noise = noise_std * np.random.randn(n_points, 1)
    y = y_true + noise

    # Store the true parameters
    true_params = {"slope": true_slope, "intercept": true_intercept}

    # Log the generated parameters
    logging.info(
        f"Generated linear data with true slope={true_slope:.2f}, intercept={true_intercept:.2f}"
    )

    # Return the generated data, the true function, and true parameters
    return X, y, true_func, true_params


def generate_polynomial_data(
    n_points: int = N_POINTS,
    x_min: float = X_MIN,
    x_max: float = X_MAX,
    noise_std: float = NOISE_STD_POLY,
) -> Tuple[
    np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], Dict[str, float]
]:
    """
    Generates data following a quadratic trend (y = ax^2 + bx + c) with added random noise.

    Args: (similar to generate_linear_data)
        ...

    Returns:
        A tuple containing:
            - X: The input feature array.
            - y: The target variable array.
            - true_func: The true underlying quadratic function.
            - true_params: Dictionary with the true coefficients a, b, c.
    """

    # Generate X values similarly to the linear case
    X = (x_max - x_min) * np.random.rand(n_points, 1) + x_min

    # Define the true parameters for the quadratic equation (y = ax^2 + bx + c)
    a = np.random.uniform(-1, 1)  # Coefficient for x^2 term
    b = np.random.uniform(-3, 3)  # Coefficient for x term
    c = np.random.uniform(0, 5)  # Constant term (intercept)

    # Define the true quadratic function using a lambda
    true_func = lambda x: a * x**2 + b * x + c

    # Calculate the true y values
    y_true = true_func(X)

    # Add Gaussian noise
    noise = noise_std * np.random.randn(n_points, 1)
    y = y_true + noise

    # Store true parameters
    true_params = {"a": a, "b": b, "c": c}
    logging.info(
        f"Generated polynomial data with true params a={a:.2f}, b={b:.2f}, c={c:.2f}"
    )

    # Return the generated data, true function, and parameters
    return X, y, true_func, true_params


def generate_tree_data(
    n_points: int = N_POINTS,
    x_min: float = X_MIN,
    x_max: float = X_MAX,
    noise_std: float = NOISE_STD_TREE,
) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], Dict]:
    """
    Generates data with a more complex, non-linear pattern (sine wave + step function)
    suitable for a decision tree model, with added random noise.

    Args: (similar to generate_linear_data)

    Returns:
        A tuple containing:
            - X: The input feature array.
            - y: The target variable array.
            - true_func: The true underlying complex function.
            - true_params: An empty dictionary (no simple parameters to represent this function).
    """

    # Generate X values
    X = (x_max - x_min) * np.random.rand(n_points, 1) + x_min

    # Define a more complex true function: a sine wave plus a step function
    # (X > 2.5).astype(float) creates an array of 0s and 1s (0 where X<=2.5, 1 where X>2.5)
    # .flatten() is sometimes needed to ensure compatible shapes for broadcasting/plotting
    true_func = (
        lambda x: np.sin(x * 1.5).flatten() + (x > 2.5).astype(float).flatten() * 2
    )

    # Calculate the true y values based on this function
    # We need to reshape the output of true_func to be a column vector [n_points, 1] like X
    y_true = true_func(X).reshape(-1, 1)

    # Add Gaussian noise
    noise = noise_std * np.random.randn(n_points, 1)
    y = y_true + noise

    logging.info("Generated non-linear data suitable for decision tree.")

    # Return generated data, true function. No simple params to return.
    return X, y, true_func, {}


# --- Model Training and Evaluation Functions ---
# These functions handle the training, (optional) tuning, and evaluation of each model type.


def train_evaluate_linear(
    X_train: np.ndarray,  # Input features for training
    y_train: np.ndarray,  # Target variable for training
    X_test: np.ndarray,  # Input features for testing
    y_test: np.ndarray,  # Target variable for testing
) -> Dict[str, Any]:  # Returns a dictionary containing results and the model
    """
    Trains a standard Linear Regression model and evaluates it on the test set.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.

    Returns:
        A dictionary containing the trained model, test scores (R², MSE),
        and fitted parameters (slope, intercept).
    """
    logging.info("[1] Linear regression training")

    model = LinearRegression()

    # Train the model using the training data.
    # .fit() finds the optimal slope and intercept that minimize the error
    # between the model's predictions and the actual y_train values.
    model.fit(X_train, y_train)

    # Make predictions on the unseen test data
    y_pred_test = model.predict(X_test)

    # Evaluate the model's performance on the test set
    # r2_score: Coefficient of determination. Ranges from -inf to 1. 1 is perfect fit.
    #           0 means the model is no better than predicting the mean. Negative means worse.
    r2_test = r2_score(y_test, y_pred_test)
    # mean_squared_error: Average of the squared differences between predicted and actual values.
    #                     Lower is better. Sensitive to outliers.
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Log the fitted parameters and test scores
    # .coef_ contains the slope(s), .intercept_ contains the intercept
    logging.info(
        f"   Fitted slope: {model.coef_[0][0]:.2f}, intercept: {model.intercept_[0]:.2f}"
    )
    logging.info(f"   Test R²: {r2_test:.3f}, Test MSE: {mse_test:.3f}")

    # Return a dictionary summarizing the results
    return {
        "model_type": "Linear",  # Identifier for this model type
        "model": model,  # The trained scikit-learn model object
        "r2_test": r2_test,  # R-squared score on the test set
        "mse_test": mse_test,  # Mean Squared Error on the test set
        "params": {
            "slope": model.coef_[0][0],
            "intercept": model.intercept_[0],
        },  # Fitted params
    }


def tune_evaluate_polynomial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    degrees: List[int],  # List of polynomial degrees to try
    cv_folds: int = CV_FOLDS,  # Number of cross-validation folds
) -> Dict[str, Any]:
    """
    Tunes the degree of a Polynomial Regression model using GridSearchCV
    (Cross-Validation) and evaluates the best found model on the test set.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        degrees: List of polynomial degrees to test (hyperparameter).
        cv_folds: Number of folds for cross-validation.

    Returns:
        A dictionary containing the best model found, test scores, best parameters,
        CV results, and the best CV score.
    """
    logging.info("[2] Polynomial regression tuning (using GridSearchCV)")

    # Create a scikit-learn Pipeline. This chains steps together.
    # Why Pipeline? When using cross-validation with feature generation (like PolynomialFeatures),
    # the feature generation MUST happen *inside* each fold to avoid data leakage.
    # The pipeline ensures PolynomialFeatures is applied only to the training part of each fold.
    pipeline = Pipeline(
        [
            # Step 1: 'poly_features' - Create polynomial features (e.g., x -> [x, x^2])
            # include_bias=False: The LinearRegression step will handle the intercept (bias) term.
            ("poly_features", PolynomialFeatures(include_bias=False)),
            # Step 2: 'lin_reg' - Apply linear regression to the transformed features.
            ("lin_reg", LinearRegression()),
        ]
    )

    # Define the hyperparameter grid to search.
    # The key format is '<pipeline_step_name>__<parameter_name>'.
    # We want to tune the 'degree' parameter of the 'poly_features' step.
    param_grid = {"poly_features__degree": degrees}

    # Set up GridSearchCV. This performs an exhaustive search over the specified parameter grid.
    # For each combination of parameters:
    #   1. It splits the *training data* (X_train, y_train) into `cv_folds` folds.
    #   2. It trains the pipeline on `cv_folds - 1` folds and evaluates on the remaining fold.
    #   3. It repeats this so each fold is used as the evaluation set once.
    #   4. It averages the evaluation scores (here, 'r2') across the folds.
    # estimator: The model or pipeline to tune.
    # param_grid: The dictionary of parameters to try.
    # cv: Number of cross-validation folds.
    # scoring='r2': The metric used to evaluate performance within CV (higher is better for R²).
    # n_jobs=-1: Use all available CPU cores to speed up the search.
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv_folds, scoring="r2", n_jobs=-1
    )

    # Run the grid search on the training data. This finds the best hyperparameters.
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameter combination found during cross-validation
    best_degree = grid_search.best_params_["poly_features__degree"]
    # Get the average R² score achieved by the best parameters during cross-validation
    best_cv_score = grid_search.best_score_
    # Get the pipeline refitted on the *entire* training set using the best parameters found
    best_model = grid_search.best_estimator_

    # --- Final Evaluation on the Test Set ---
    # Now, evaluate the single *best* model (chosen using only training data via CV)
    # on the completely held-out test set to get an unbiased estimate of its generalization performance.
    y_pred_test = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Log the tuning results and the final test evaluation
    logging.info(f"   Best degree found via CV: {best_degree}")
    logging.info(
        f"   Best CV R² score (average R² across folds on training data): {best_cv_score:.3f}"
    )
    logging.info(
        f"   Test R² (using best model on unseen test data): {r2_test:.3f}, Test MSE: {mse_test:.3f}"
    )

    # Return results, including CV details for potential analysis
    return {
        "model_type": "Polynomial",
        "model": best_model,  # The final model trained with the best degree
        "r2_test": r2_test,  # Final R² score on the test set
        "mse_test": mse_test,  # Final MSE on the test set
        "best_params": grid_search.best_params_,  # Dictionary of best hyperparameters found
        "cv_results": grid_search.cv_results_,  # Detailed results from cross-validation
        "best_cv_score": best_cv_score,  # The mean CV score of the best estimator
    }


def tune_evaluate_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    depths: List[int],  # List of max_depth values to try
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,  # Seed for the tree regressor
) -> Dict[str, Any]:
    """
    Tunes the max_depth of a Decision Tree Regressor using GridSearchCV
    and evaluates the best found model on the test set.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        depths: List of max_depth values to test (hyperparameter).
        cv_folds: Number of folds for cross-validation.
        random_state: Seed for the DecisionTreeRegressor for reproducibility.

    Returns:
        A dictionary containing the best model found, test scores, best parameters,
        CV results, and the best CV score.
    """
    logging.info("[3] Decision tree regression tuning (using GridSearchCV)")

    # Create an instance of the DecisionTreeRegressor.
    # random_state ensures that the tree building process is deterministic (given the same data).
    tree_reg = DecisionTreeRegressor(random_state=random_state)

    # Define the hyperparameter grid: we want to tune 'max_depth'.
    # max_depth controls how deep the tree can grow. Deeper trees can model more complex
    # patterns but are more prone to overfitting the training data.
    param_grid = {"max_depth": depths}

    # Set up GridSearchCV, similar to the polynomial case.
    # Here, the 'estimator' is the DecisionTreeRegressor itself (no pipeline needed as
    # there's no feature transformation step that depends on the data split).
    grid_search = GridSearchCV(
        tree_reg, param_grid, cv=cv_folds, scoring="r2", n_jobs=-1
    )

    # Run the grid search on the training data.
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameter (max_depth) and corresponding CV score
    best_depth = grid_search.best_params_["max_depth"]
    best_cv_score = grid_search.best_score_
    # Get the tree refitted on the whole training set with the best depth
    best_model = grid_search.best_estimator_

    # --- Final Evaluation on the Test Set ---
    # Evaluate the best tree on the held-out test set.
    y_pred_test = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Log the results
    logging.info(f"   Best depth found via CV: {best_depth}")
    logging.info(f"   Best CV R² score: {best_cv_score:.3f}")
    logging.info(
        f"   Test R² (using best model): {r2_test:.3f}, Test MSE: {mse_test:.3f}"
    )

    # Return results
    return {
        "model_type": "Decision Tree",
        "model": best_model,
        "r2_test": r2_test,
        "mse_test": mse_test,
        "best_params": grid_search.best_params_,
        "cv_results": grid_search.cv_results_,
        "best_cv_score": best_cv_score,
    }


# --- Plotting Function ---


def plot_results(
    results: List[
        Dict[str, Any]
    ],  # List of result dictionaries from the evaluation functions
    datasets: Dict[
        str, Dict[str, np.ndarray]
    ],  # Dict mapping model type to its train/test data & true func
    plot_config: Dict,  # Dictionary with plotting range and resolution
):
    """
    Generates subplots comparing each model's predictions against the data.

    Args:
        results: A list where each element is a result dictionary from one of the
                 train/evaluate/tune functions.
        datasets: A dictionary where keys are model types ('Linear', 'Polynomial', 'Decision Tree')
                  and values are dictionaries holding the corresponding 'X_train', 'y_train',
                  'X_test', 'y_test', and potentially 'true_func'.
        plot_config: Contains plot settings like x-axis range and number of points.
    """
    logging.info("=== Plotting results ===")
    n_models = len(results)  # Number of models to plot

    # Create a figure and a set of subplots.
    # nrows=n_models, ncols=1: Arrange plots vertically.
    # figsize: Size of the entire figure in inches.
    # sharex=True: All subplots will share the same x-axis range and ticks.
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 6 * n_models), sharex=True)

    # If there's only one plot, plt.subplots returns a single Axes object, not an array.
    # Make it a list/array so we can iterate over it consistently.
    if n_models == 1:
        axes = [axes]

    # Generate a dense range of X values for plotting smooth model prediction lines
    x_plot = np.linspace(
        plot_config["x_min"], plot_config["x_max"], plot_config["n_plot_points"]
    ).reshape(-1, 1)

    # Define plot styles for each model type
    plot_styles = {
        "Linear": {"color": "red", "data_color": "blue"},
        "Polynomial": {"color": "green", "data_color": "purple"},
        "Decision Tree": {"color": "black", "data_color": "orange"},
    }

    # Loop through each model's results and corresponding dataset
    for i, result in enumerate(results):
        model_type = result["model_type"]  # e.g., "Linear", "Polynomial"
        model = result["model"]  # The trained scikit-learn model object
        r2_test = result["r2_test"]  # Test R² score
        mse_test = result["mse_test"]  # Test MSE score
        data = datasets[model_type]  # Get the specific dataset for this model type
        style = plot_styles[model_type]  # Get the plotting colors

        # Use the final trained/tuned model to predict y values for the smooth x_plot range
        y_plot = model.predict(x_plot)

        # --- Plotting Data Points ---
        # Plot training data points: lighter color, smaller size
        axes[i].scatter(
            data["X_train"],
            data["y_train"],
            color=style["data_color"],
            alpha=0.3,
            s=20,
            label="Train data",
        )
        # Plot test data points: darker color, larger size
        axes[i].scatter(
            data["X_test"],
            data["y_test"],
            color=style["data_color"],
            alpha=0.8,
            s=40,
            label="Test data",
        )

        # --- Plotting Model Prediction Line ---
        # Construct the label for the model line, including hyperparameters if applicable
        label = f"{model_type} model"
        if model_type == "Polynomial":
            # Access the best degree from the results dictionary
            label += f" (deg={result['best_params']['poly_features__degree']})"
        elif model_type == "Decision Tree":
            # Access the best depth from the results dictionary
            label += f" (depth={result['best_params']['max_depth']})"
        axes[i].plot(x_plot, y_plot, color=style["color"], linewidth=2, label=label)

        # --- Plotting True Function (Optional) ---
        # Check if the true underlying function is available in the dataset dictionary
        if data.get("true_func"):
            true_func = data["true_func"]
            # Calculate the true y values for the plotting range
            y_true_plot = true_func(x_plot)
            # Ensure y_true_plot is a 1D array for plotting if needed
            # (Some true_func might return column vectors, others might return 1D)
            if y_true_plot.ndim > 1 and y_true_plot.shape[1] == 1:
                y_true_plot = y_true_plot.flatten()
            # Plot the true function as a dashed gray line
            axes[i].plot(
                x_plot,
                y_true_plot,
                color="gray",
                linestyle="--",
                linewidth=2,
                label="True function",
            )

        # --- Plot Configuration ---
        # Set the title for the subplot, including key performance metrics
        title = (
            f"{model_type} Regression (Test R² = {r2_test:.3f}, MSE = {mse_test:.3f})"
        )
        # Add the best Cross-Validation score to the title if available (for tuned models)
        if "best_cv_score" in result:
            title += f"\nBest CV R² = {result['best_cv_score']:.3f} (on train folds)"  # Clarify CV score context

        axes[i].set_title(title)  # Set the title
        axes[i].set_ylabel("y")  # Set the y-axis label
        axes[
            i
        ].legend()  # Display the legend (shows labels for plots and scatter points)
        axes[i].grid(True)  # Add a grid to the plot for better readability

    # Set the x-axis label only on the bottom-most plot (since axes are shared)
    axes[-1].set_xlabel("X")
    # Adjust plot layout to prevent labels/titles from overlapping
    plt.tight_layout()
    # Optionally save the figure to a file
    # output_dir = Path(__file__).parent
    # plt.savefig(output_dir / "regression_comparison.png", dpi=300)
    # Display the plot window
    plt.savefig("ml.png")


# --- Main Execution ---


def main():
    """
    Main function to orchestrate the data generation, model training/tuning,
    evaluation, and plotting process.
    """
    logging.info("=== Generating datasets and preparing for modeling ===")

    # Dictionary to store the datasets for each model type
    datasets = {}

    # --- Generate and Split Data for Each Model ---

    # 1. Linear Data
    X_lin, y_lin, true_func_lin, _ = generate_linear_data()
    # Split the data into training and testing sets.
    # test_size: Proportion of data for the test set.
    # random_state: Ensures the split is the same every time (for reproducibility).
    # Stratify is not typically used for regression splits.
    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Store the split data and the true function in the datasets dictionary
    datasets["Linear"] = {
        "X_train": X_train_lin,
        "y_train": y_train_lin,
        "X_test": X_test_lin,
        "y_test": y_test_lin,
        "true_func": true_func_lin,  # Store the function itself
    }

    # 2. Polynomial Data
    X_poly, y_poly, true_func_poly, _ = generate_polynomial_data()
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
        X_poly, y_poly, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    datasets["Polynomial"] = {
        "X_train": X_train_poly,
        "y_train": y_train_poly,
        "X_test": X_test_poly,
        "y_test": y_test_poly,
        "true_func": true_func_poly,
    }

    # 3. Decision Tree Data (non-linear, step-like)
    X_tree, y_tree, true_func_tree, _ = generate_tree_data()
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
        X_tree, y_tree, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    datasets["Decision Tree"] = {
        "X_train": X_train_tree,
        "y_train": y_train_tree,
        "X_test": X_test_tree,
        "y_test": y_test_tree,
        "true_func": true_func_tree,
    }

    logging.info("\n=== Training and Evaluating Models ===")
    # List to store the result dictionaries from each model evaluation
    results = []

    # --- Train/Tune and Evaluate Each Model ---

    # 1. Linear Model (Simple Train/Evaluate)
    lin_results = train_evaluate_linear(
        X_train_lin, y_train_lin, X_test_lin, y_test_lin
    )
    results.append(lin_results)  # Add results to the list

    # 2. Polynomial Model (Tune Degree using CV, then Evaluate)
    poly_results = tune_evaluate_polynomial(
        X_train_poly,
        y_train_poly,
        X_test_poly,
        y_test_poly,
        degrees=POLYNOMIAL_DEGREES_TO_TRY,  # Pass the list of degrees to try
        cv_folds=CV_FOLDS,  # Pass the number of CV folds
    )
    results.append(poly_results)

    # 3. Decision Tree Model (Tune Depth using CV, then Evaluate)
    tree_results = tune_evaluate_tree(
        X_train_tree,
        y_train_tree,
        X_test_tree,
        y_test_tree,
        depths=TREE_DEPTHS_TO_TRY,  # Pass the list of depths to try
        cv_folds=CV_FOLDS,  # Pass the number of CV folds
        random_state=RANDOM_STATE,  # Pass the random state for reproducibility
    )
    results.append(tree_results)

    # --- Plotting ---
    plot_config = {"x_min": X_MIN, "x_max": X_MAX, "n_plot_points": N_PLOT_POINTS}
    plot_results(results, datasets, plot_config)

    logging.info("\n=== Script execution finished ===")


if __name__ == "__main__":
    main()
