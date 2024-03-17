""" 
function [ ft ] = Objfun (x)
    ft = sum ( abs(x) ) + prod( abs(x) );
end
"""

import numpy as np

def Objfun(x):
    ft = np.sum(abs(x)) + np.prod(abs(x))
    return ft