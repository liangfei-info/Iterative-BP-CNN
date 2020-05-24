# This file is to generate the matrix for generating colored noise.
import numpy as np
from numpy import linalg as la
import struct


eta = 0.5
N = 1998
cov = np.zeros((N, N))
for i in range(N):
    for j in range(i,N):
        cov[i, j] = eta**(abs(i - j))
        cov[j, i] = cov[i, j]
# v 为特征值    P为特征向量
v, P = la.eig(cov)
V = np.diag(v**(0.5))
transfer_mat = P @ V @ la.inv(P)
fout_file = format('cov_1_2_corr_para%.2f.dat' % (eta))

#transfer_mat.tofile(fout_file)
# with open(fout_file, 'wb') as fout:#这两种写文件的方法得到的结果是完全一样的
#     fout.write(transfer_mat)
with open(fout_file, 'wb') as fout:
    # np.array(transfer_mat.shape).tofile(f)
    transfer_mat.astype(np.float32).T.tofile(fout)
fout.close()
