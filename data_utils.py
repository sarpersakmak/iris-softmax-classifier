"""
Title:       data_utils.py
Author:      Sarper Sakmak
ID:          14008175400
Section:     1
Assignment:  CMPE 442 Programming Assignment-1
Description:
    - Loads the Iris dataset using scikit-learn.
    - Splits the dataset into training (70%), validation (15%), and test (15%) sets
      using two sequential stratified train_test_split calls with random_state=42.
    - Constructs three polynomial feature representations:
        * Degree 1 (Linear)    : 4 features  -> {x_i}
        * Degree 2 (Quadratic) : 15 features -> {x_i, x_i^2, x_i*x_j}
        * Degree 3 (Cubic)     : 35 features -> {x_i, x_i^2, x_i*x_j, x_i^3, x_i^2*x_j, x_i*x_j*x_k}
    - PolynomialFeatures is fitted ONLY on the training set to prevent data leakage.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Fix random seed for reproducibility
np.random.seed(42)


def load_and_split_data():
    """
    STEP 1: Load the Iris dataset and split into train / val / test sets.

    Split strategy:
        - First  split : 70% train, 30% temp  (val + test pool)
        - Second split : 50% of temp -> val  (15% total)
                         50% of temp -> test (15% total)
    Stratified splitting ensures balanced class distribution in all subsets.
    """
    iris = load_iris()
    X, y = iris.data, iris.target   # X: (150, 4)  |  y: (150,)

    # First split: separate training set from the temporary hold-out pool
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Second split: divide the hold-out pool equally into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[data_utils] Dataset split -> "
          f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def expand_features(X_train, X_val, X_test, degree):
    """
    STEP 2: Expand raw features to the requested polynomial degree.

    Degree 1 -> 4  features : original linear terms only
    Degree 2 -> 15 features : linear + squared + pairwise products
    Degree 3 -> 35 features : degree-2 terms + cubic + mixed cubic terms

    Parameters
    ----------
    X_train, X_val, X_test : numpy arrays with raw features
    degree                 : integer (1, 2, or 3)

    Returns
    -------
    X_train_poly, X_val_poly, X_test_poly : transformed numpy arrays
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    # Fit ONLY on training data; transform val/test without re-fitting
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly   = poly.transform(X_val)
    X_test_poly  = poly.transform(X_test)

    print(f"[data_utils] Degree {degree} -> {X_train_poly.shape[1]} features")

    return X_train_poly, X_val_poly, X_test_poly


def get_all_polynomial_data():
    """
    STEP 3: Build and return all three polynomial representations of the dataset.

    Returns
    -------
    data : dict  { degree -> (X_train_poly, X_val_poly, X_test_poly,
                              y_train, y_val, y_test) }
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    data = {}
    for degree in [1, 2, 3]:
        X_tr_p, X_v_p, X_te_p = expand_features(X_train, X_val, X_test, degree)
        data[degree] = (X_tr_p, X_v_p, X_te_p, y_train, y_val, y_test)

    return data