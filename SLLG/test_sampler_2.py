import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from sample_LLG_function_noise_2 import sample_LLG_function_noise_2
from fenics import *


T = 1
FEMOrder = 1
BDFOrder = 1
Nh = 16
Ntau = Nh * 4

# param = np.random.normal(0, 1, 1000)
param = np.array([0,0,0,1])
x = sample_LLG_function_noise_2(param, Nh, Ntau, T, FEMOrder, BDFOrder)


