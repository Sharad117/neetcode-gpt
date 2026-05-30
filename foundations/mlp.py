import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        N= len(weights) 
        res=np.maximum(x@weights[0]+biases[0],0)
        for i in range(1,N):
            res=res@weights[i]+biases[i]
            res=np.maximum(res,0) 
        return np.round(res,5)


