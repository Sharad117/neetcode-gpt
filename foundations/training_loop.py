import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_features=X.shape[1]
        n_samples=X.shape[0]
        w=np.zeros((n_features,1))
        b=0.0
        y=y.reshape(-1,1)
        for _ in range(epochs):
            y_pred=X@w+b 
            grad_w= (2/n_samples) * X.T @ (y_pred-y)
            grad_b =  2* np.mean(y_pred-y)
            w=w-lr*grad_w 
            b=b-lr*grad_b 

        return (np.round(np.squeeze(w,axis=1),5),np.round(b,5))
