import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import func as f

N = 500
#gamma = 0.00001
#x1 = np.random.uniform(-1, 1, N)
#x2 = np.random.uniform(-1, 1, N)
#x3 = np.random.uniform(-1, 1, N)
#e = np.random.normal(0, gamma)
#x4 = x1 + x2 + x3 + e
#x5 = np.random.uniform(-1, 1, N)
#x6 = np.random.uniform(-1, 1, N)
#x7 = np.random.uniform(-1, 1, N)
#f.WritingInFile(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], [x1, x2, x3, x4, x5, x6, x7], 'X.txt')
x1, x2, x3, x4, x5, x6, x7 = f.get_x('X.txt')
matr_X = f.create_X_matr(x1, x2, x3, x4, x5, x6, x7)
det_XtX, XtX = f.det_inf_matr(matr_X)
min_eigvals, max_eigvals = f.eigen_vals(XtX)
cond_NG = f.measure_cond_matr_Neumann_Goldstein(min_eigvals, max_eigvals)
max_r, r = f.pair_conjugation(matr_X)
R, R_max = f.conjugation(r)
#y = f.FindResponds(x1, x2, x3, x4, x5, x6, x7,'u_y_ej_x1_x2.txt', N)
y = np.array(f.get_y('u_y_ej_x1_x2.txt'))
est_theta, norm, RSS, norm_1, lambd = f.ridge_estimation(XtX, matr_X, y)
f.Graph(lambd, norm)
f.Graph(lambd, RSS)
est_theta_1, norm_1, RSS_1 = f.estimation_PCA(matr_X, y, N)
