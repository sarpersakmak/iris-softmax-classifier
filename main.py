"""
Title:       main.py
Author:      Sarper Sakmak
ID:          14008175400
Section:     1
Assignment:  CMPE 442 Programming Assignment-1
Description:
    - Entry point for the entire experiment pipeline.
    - STEP 1 : Load Iris dataset and build all three polynomial feature sets.
    - STEP 2 : Run 3-fold cross-validation to select the best polynomial degree.
    - STEP 3 : Train all 9 model combinations (3 regularizations x 3 learning rates)
               using the selected polynomial degree.
    - STEP 4 : Generate and save the three loss curve figures.
    - STEP 5 : Print final test metrics for all 9 models and highlight the best one.

Usage:
    python main.py
"""

import torch
import numpy as np

from data_utils import get_all_polynomial_data
from trainer    import cross_validate, train_model, LR
from plot       import plot_all_losses

# Fix seeds for full reproducibility across all modules
torch.manual_seed(42)
np.random.seed(42)

# ── Experiment configuration ──────────────────────────────────────────────────
DEGREE_NAMES = {
    1: "Linear    (4  features)",
    2: "Quadratic (15 features)",
    3: "Cubic     (35 features)",
}

REG_TYPES = ["Ridge", "Lasso", "ElasticNet"]
LR_KEYS   = ["l1", "l2", "l3"]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Load dataset and create all polynomial feature representations
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: Loading Iris dataset and generating polynomial features")
print("=" * 65)

# data[degree] = (X_train_poly, X_val_poly, X_test_poly, y_train, y_val, y_test)
data = get_all_polynomial_data()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: 3-Fold Cross-Validation — select the best polynomial degree
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: 3-Fold Cross-Validation for Polynomial Degree Selection")
print("=" * 65)

cv_scores = {}
for degree in [1, 2, 3]:
    X_train_poly, _, _, y_train, _, _ = data[degree]
    input_dim     = X_train_poly.shape[1]
    mean_val_loss = cross_validate(X_train_poly, y_train, input_dim, k=3)
    cv_scores[degree] = mean_val_loss
    print(f"  Degree {degree} | {DEGREE_NAMES[degree]} | "
          f"Mean CV Val Loss = {mean_val_loss:.4f}")

# The degree with the lowest mean CV validation loss is selected
best_degree = min(cv_scores, key=cv_scores.get)
print(f"\n>>> Best model selected: {DEGREE_NAMES[best_degree]} <<<")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Train all 9 combinations on the selected polynomial features
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 3: Training 9 Models — {DEGREE_NAMES[best_degree]}")
print("=" * 65)

X_train, X_val, X_test, y_train, y_val, y_test = data[best_degree]
input_dim = X_train.shape[1]

# results dict: (reg_type, lr_key) -> loss lists + metrics
results = {}

for reg in REG_TYPES:
    for lr_key in LR_KEYS:
        label = f"{reg}-{lr_key}"
        print(f"  Training: {label:<22}  lr = {LR[lr_key]}")

        tr_l, v_l, te_l, metrics = train_model(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            input_dim=input_dim,
            reg_type=reg,
            lr_key=lr_key,
        )
        results[(reg, lr_key)] = {
            "train_losses": tr_l,
            "val_losses"  : v_l,
            "test_losses" : te_l,
            "metrics"     : metrics,
        }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Generate and save loss curve figures
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: Generating Loss Curve Figures")
print("=" * 65)

plot_all_losses(results)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Print final test metrics and highlight the best model
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5: Final Test Set Metrics for All 9 Models")
print("=" * 65)

header = f"{'Model':<22} {'Val Loss':>9} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
print(f"\n{header}")
print("-" * 65)

best_key      = None
best_val_loss = float("inf")

for (reg, lr_key), v in results.items():
    m     = v["metrics"]
    label = f"{reg}-{lr_key}"
    print(f"{label:<22} {m['val_loss']:>9.4f} {m['accuracy']:>7.4f} "
          f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")

    # Track the model with the lowest final validation loss
    if m["val_loss"] < best_val_loss:
        best_val_loss = m["val_loss"]
        best_key      = (reg, lr_key)

# Print highlighted summary for the best model
best_reg, best_lr = best_key
best_m = results[best_key]["metrics"]

print("\n" + "=" * 65)
print(f">>> Best model: {best_reg} + {best_lr}  |  Val Loss = {best_val_loss:.4f} <<<")
print(f"    Accuracy  : {best_m['accuracy']:.4f}")
print(f"    Precision : {best_m['precision']:.4f}")
print(f"    Recall    : {best_m['recall']:.4f}")
print(f"    F1-Score  : {best_m['f1']:.4f}")
print("=" * 65)