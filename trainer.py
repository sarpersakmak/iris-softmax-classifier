"""
Title:       trainer.py
Author:      Sarper Sakmak
ID:          14008175400
Section:     1
Assignment:  CMPE 442 Programming Assignment-1
Description:
    - Trains SoftmaxRegression models for 50 epochs using SGD.
    - Supports three regularization techniques added manually to CrossEntropyLoss:
        * Ridge     (L2) : lambda * sum(w^2)
        * Lasso     (L1) : lambda * sum(|w|)
        * ElasticNet      : lambda * (0.5*L1 + 0.5*L2)  [l1_ratio = 0.5]
    - Three learning rates within the assignment-specified intervals:
        l1 = 0.00001   (range: 0.00001 <= l1 <= 0.00002)
        l2 = 0.0015    (range: 0.001   <= l2 <= 0.002  )
        l3 = 0.15      (range: 0.1     <= l3 <= 0.2    )
    - Records train_loss, val_loss, and test_loss for every epoch.
    - Computes final accuracy, precision, recall, and F1-score (macro) on the test set.
    - Provides a 3-fold cross-validation function for polynomial degree selection.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import SoftmaxRegression

# Fix seeds for full reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters (values not explicitly specified -> chosen freely per Note 2) ──
EPOCHS      = 50       # number of training epochs
LAMBDA_REG  = 1e-3     # regularization strength lambda
L1_RATIO    = 0.5      # ElasticNet mixing ratio (equal weight to L1 and L2)
NUM_CLASSES = 3        # Iris has 3 classes

# Three learning rates, one from each required interval
LR = {
    "l1": 0.00001,   # slowest  – tests convergence under minimal updates
    "l2": 0.0015,    # moderate – expected best balance
    "l3": 0.15,      # fastest  – tests aggressive convergence
}


# ── Helper: convert numpy array to PyTorch tensor ────────────────────────────
def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype)


# ── Regularization penalty ───────────────────────────────────────────────────
def regularization_loss(model, reg_type):
    """
    STEP A: Compute the regularization penalty for the weight parameter.

    Ridge     : lambda * sum(w_i^2)
    Lasso     : lambda * sum(|w_i|)
    ElasticNet: lambda * (l1_ratio * sum(|w_i|) + (1-l1_ratio) * sum(w_i^2))

    Parameters
    ----------
    model    : SoftmaxRegression instance
    reg_type : str  "Ridge" | "Lasso" | "ElasticNet"

    Returns
    -------
    Scalar tensor representing the regularization term.
    """
    w = model.weight

    if reg_type == "Ridge":
        return LAMBDA_REG * torch.sum(w ** 2)

    elif reg_type == "Lasso":
        return LAMBDA_REG * torch.sum(torch.abs(w))

    elif reg_type == "ElasticNet":
        l1_term = torch.sum(torch.abs(w))
        l2_term = torch.sum(w ** 2)
        return LAMBDA_REG * (L1_RATIO * l1_term + (1.0 - L1_RATIO) * l2_term)

    else:
        raise ValueError(f"[trainer] Unknown regularization type: {reg_type}")


# ── Main training function ───────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                input_dim, reg_type, lr_key):
    """
    STEP B: Train one SoftmaxRegression model for EPOCHS epochs.

    Total loss = CrossEntropyLoss + regularization_loss

    Parameters
    ----------
    X_train / y_train : training features and labels (numpy)
    X_val   / y_val   : validation features and labels (numpy)
    X_test  / y_test  : test features and labels (numpy)
    input_dim         : number of input features
    reg_type          : "Ridge" | "Lasso" | "ElasticNet"
    lr_key            : "l1" | "l2" | "l3"

    Returns
    -------
    train_losses, val_losses, test_losses : lists of float (length = EPOCHS)
    metrics : dict with accuracy, precision, recall, f1, val_loss (final epoch)
    """
    lr = LR[lr_key]

    # STEP B1: Convert numpy arrays to PyTorch tensors
    X_tr = to_tensor(X_train)
    y_tr = to_tensor(y_train, dtype=torch.long)
    X_v  = to_tensor(X_val)
    y_v  = to_tensor(y_val,   dtype=torch.long)
    X_te = to_tensor(X_test)
    y_te = to_tensor(y_test,  dtype=torch.long)

    # STEP B2: Instantiate a fresh model for each combination
    torch.manual_seed(42)
    model     = SoftmaxRegression(input_dim=input_dim, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses, val_losses, test_losses = [], [], []

    # STEP B3: Training loop over 50 epochs
    for epoch in range(EPOCHS):

        # ── Forward + backward pass on training set ──
        model.train()
        optimizer.zero_grad()

        logits_tr  = model(X_tr)
        ce_loss    = criterion(logits_tr, y_tr)
        reg_loss   = regularization_loss(model, reg_type)
        total_loss = ce_loss + reg_loss   # combined loss function

        total_loss.backward()    # compute gradients
        optimizer.step()         # update parameters

        # ── Evaluate on validation and test sets (no gradients needed) ──
        model.eval()
        with torch.no_grad():
            # Validation loss
            v_loss = (criterion(model(X_v), y_v)
                      + regularization_loss(model, reg_type))

            # Test loss
            te_loss = (criterion(model(X_te), y_te)
                       + regularization_loss(model, reg_type))

        # Record losses for this epoch
        train_losses.append(total_loss.item())
        val_losses.append(v_loss.item())
        test_losses.append(te_loss.item())

    # STEP B4: Compute final classification metrics on test set
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_te), dim=1).numpy()

    metrics = {
        "accuracy" : accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall"   : recall_score(y_test, preds, average="macro", zero_division=0),
        "f1"       : f1_score(y_test, preds, average="macro", zero_division=0),
        "val_loss" : val_losses[-1],   # final epoch validation loss
    }

    return train_losses, val_losses, test_losses, metrics


# ── 3-Fold Cross-Validation for polynomial degree selection ──────────────────
def cross_validate(X_train_poly, y_train, input_dim, k=3):
    """
    STEP C: Perform k-fold cross-validation on the training set.

    Uses Ridge regularization with l2 learning rate as a representative config.
    The degree yielding the lowest mean validation loss will be selected.

    Parameters
    ----------
    X_train_poly : polynomial-expanded training features (numpy)
    y_train      : training labels (numpy)
    input_dim    : number of features in the expanded representation
    k            : number of folds (default 3 as required by the assignment)

    Returns
    -------
    mean_val_loss : float  – average validation loss across k folds
    """
    n         = len(X_train_poly)
    fold_size = n // k

    # Shuffle indices once for unbiased fold assignment
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    val_losses_cv = []

    for fold in range(k):
        # STEP C1: Create train / validation index split for this fold
        val_idx   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([
            indices[: fold * fold_size],
            indices[(fold + 1) * fold_size :]
        ])

        X_tr_cv = X_train_poly[train_idx]
        y_tr_cv = y_train[train_idx]
        X_v_cv  = X_train_poly[val_idx]
        y_v_cv  = y_train[val_idx]

        # STEP C2: Convert to tensors
        X_tr_t = to_tensor(X_tr_cv)
        y_tr_t = to_tensor(y_tr_cv, dtype=torch.long)
        X_v_t  = to_tensor(X_v_cv)
        y_v_t  = to_tensor(y_v_cv,  dtype=torch.long)

        # STEP C3: Train a temporary model on this fold
        torch.manual_seed(42)
        model     = SoftmaxRegression(input_dim=input_dim, num_classes=NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR["l2"])

        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = (criterion(model(X_tr_t), y_tr_t)
                    + regularization_loss(model, "Ridge"))
            loss.backward()
            optimizer.step()

        # STEP C4: Record the validation loss for this fold
        model.eval()
        with torch.no_grad():
            fold_val_loss = criterion(model(X_v_t), y_v_t).item()
        val_losses_cv.append(fold_val_loss)

    return float(np.mean(val_losses_cv))