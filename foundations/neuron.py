import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self,z):
        return np.round(1/(1+np.exp(-z)),5)
    def relu(self,z):
        res=np.maximum(0,z)
        return np.round(res,5)
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
        z= np.dot(w,x)+b 
        if activation=='sigmoid':
            return self.sigmoid(z)
        elif activation=='relu':
            return self.relu(z)


