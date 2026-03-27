"""
Title:       model.py
Author:      Sarper Sakmak
ID:          14008175400
Section:     1
Assignment:  CMPE 442 Programming Assignment-1
Description:
    - Defines the SoftmaxRegression model as a torch.nn.Module subclass.
    - Weight matrix W and bias vector b are declared as torch.nn.Parameter objects.
    - nn.Linear is NOT used anywhere in this file (prohibited by the assignment).
    - The forward pass is computed exclusively with torch.matmul:
          logits = X @ W + b
    - CrossEntropyLoss (called in trainer.py) applies log-softmax internally,
      so the forward method returns raw logits.
"""

import torch
import torch.nn as nn

# Fix random seed so weight initialisation is reproducible
torch.manual_seed(42)


class SoftmaxRegression(nn.Module):
    """
    Softmax Regression implemented with raw trainable Parameters.

    Architecture
    ------------
    Weight : nn.Parameter  shape (input_dim, num_classes)   Xavier-like init
    Bias   : nn.Parameter  shape (num_classes,)             zero init
    Output : logits of shape (batch_size, num_classes)
    """

    def __init__(self, input_dim, num_classes=3):
        """
        STEP 1: Initialise trainable parameters.

        Parameters
        ----------
        input_dim   : number of input features (4 / 15 / 35 depending on degree)
        num_classes : number of output classes (3 for Iris)
        """
        super(SoftmaxRegression, self).__init__()

        # Weight matrix: small random values for stable gradient flow at epoch 1
        self.weight = nn.Parameter(
            torch.randn(input_dim, num_classes) * 0.01
        )

        # Bias vector: initialised to zero
        self.bias = nn.Parameter(
            torch.zeros(num_classes)
        )

    def forward(self, x):
        """
        STEP 2: Compute the forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)

        Returns
        -------
        logits : Tensor of shape (batch_size, num_classes)
                 CrossEntropyLoss handles softmax; no activation is applied here.
        """
        # Linear transformation using matrix multiplication (nn.Linear is forbidden)
        logits = torch.matmul(x, self.weight) + self.bias
        return logits