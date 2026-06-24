import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z = np.dot(x, w) + b
        activated = np.copy(z)
        if activation in ["Sigmoid", "sigmoid"]:
            activated = 1 / (1 + np.exp(activated * -1))
        elif activation in ["Relu", "relu"]:
            activated = max(0.0, activated)
        else:
            raise ValueError("should be sigmoid or relu")
        return np.round(activated, 5)
