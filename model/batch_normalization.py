import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists


        if training:
            curr_mean=np.mean(x,axis=0)
            curr_var=np.var(x,axis=0)


            running_mean=(1-momentum)*np.array(running_mean)+ momentum*curr_mean 
            running_var=(1-momentum)*np.array(running_var)+momentum*curr_var 
        else:
            curr_mean=np.array(running_mean )
            curr_var=np.array(running_var )

        norm_x=(np.array(x)-curr_mean)/np.sqrt(curr_var+1e-5)
        y= gamma*norm_x+beta
        return (list(np.round(y,4)),list(np.round(running_mean,4)),list(np.round(running_var,4)))


