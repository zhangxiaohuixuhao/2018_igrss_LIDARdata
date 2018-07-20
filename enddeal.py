import numpy as np

alltest_y3 = np.load("Result/alltest_y3.npy")
alltest_y17 = np.load("Result/alltest_y17.npy")
labeltwo = np.load("result/labeltwo.npy")
endlab = np.zeros((2404, 8344))
for i in range(2404):
    for j in range(4768):
        if labeltwo[i,j] == 1:
            endlab[i,j] = alltest_y3[i,j]
        elif labeltwo[i,j] == 2:
            endlab[i, j] = alltest_y17[i, j]
np.save('result/endlab.npy',endlab)