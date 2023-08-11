import numpy as np
import matplotlib.pyplot as plt


for i in range(32):
    A = np.loadtxt("data_switch/switch_rectange_min_max_sample_"+str(i)+".csv", delimiter=",")
    tt = A[:, 0]
    min = A[:, 2]
    max = A[:, 3]
    # plt.plot(tt, min)
    plt.plot(tt, max)
plt.show()