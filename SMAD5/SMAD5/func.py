import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st

###########################
def WritingInFile(names, sequences, fileName):
    with open(fileName, 'w') as f:
        for i in range(len(names)):
            f.write(names[i] + ': ')
        f.write('\n')
        for j in range(len(sequences[0])):
            for i in range(len(names)):
                f.write(str(sequences[i][j]) + ' ')
            f.write('\n')

#############################
def get_x(fname):
    str_file = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(1, len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        x1_el, x2_el, x3_el, x4_el, x5_el, x6_el, x7_el = s.split(' ')
        x1.append(float(x1_el))
        x2.append(float(x2_el))
        x3.append(float(x3_el))
        x4.append(float(x4_el))
        x5.append(float(x5_el))
        x6.append(float(x6_el))
        x7.append(float(x7_el))
    return x1, x2, x3, x4, x5, x6, x7
#################################
def get_y(fname):
    str_file = []
    y = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(1, len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        u, ej, y_el = s.split(' ')
        y.append(float(y_el))
    return y
###############################
def create_X_matr(x1, x2, x3, x4, x5, x6, x7):
    X = [[el1, el2, el3, el4, el5, el6, el7 ] for el1, el2, el3, el4, el5, el6, el7 in zip(x1, x2, x3, x4, x5, x6, x7)]
    return np.array(X, dtype=float)
###################################3
def det_inf_matr(matr_X):
    XtX = np.matmul(matr_X.T, matr_X)
    det_XtX = np.linalg.det(XtX)
    return det_XtX, XtX
###################################
def eigen_vals(XtX):
    eig = np.linalg.eig(XtX)
    max_eig = max(eig[0])
    min_eig = min(eig[0])
    return min_eig, max_eig
#####################################
def measure_cond_matr_Neumann_Goldstein(min_eig, max_eig):
    return max_eig / min_eig
#######################################
def pair_conjugation(matr_X):
    matr_Xt = matr_X.T
    a = np.array([el for el in matr_Xt], dtype = float)
    r = []
    r_1 = []
    for i in range(7):
        r.append([])
        for j in range(7):
            if(i != j):
                r[i].append(np.sum(a[i] * a[j])/((np.linalg.norm(a[i]) * np.linalg.norm(a[j]))))
                r_1.append(np.sum(a[i] * a[j])/((np.linalg.norm(a[i]) * np.linalg.norm(a[j]))))
            else:
                r[i].append(1.)
    max_r = np.max(r_1)
    return max_r, r
###################################
def conjugation(r):
    r = np.array(r)
    R = np.linalg.inv(r)
    R_ii = np.diag(R)
    R_i_2 = [1. - 1. / el for el in R_ii]
    max_R_i = np.max(R_i_2)
    return R_i_2, max_R_i
#################################
def FindResponds(x1, x2, x3, x4, x5, x6, x7, outputFile, N):
    p = 0.05
    U_1 = np.array([el1 + el2 + el3 + el4 + el5 + el6 + el7  for el1, el2, el3, el4, el5, el6, el7 in zip(x1, x2, x3, x4, x5, x6, x7)])
    mean_1 = np.sum(U_1) / N
    w2 = np.sum([(el - mean_1) ** 2 for el in U_1])
    w_2 = w2 / (N - 1)
    sigm = math.sqrt(p * w_2)
    ej = np.random.normal(0, sigm, N)  
    y = np.array([el1 + el2 for el1, el2 in zip(U_1, ej)])
    WritingInFile(['U', 'ej', 'y'], [U_1, ej, y], outputFile)
    return y
#####################################
def ridge_estimation(XtX, matr_X, y):
    lambd = [0.005, 0.01, 0.015, 0.02, 0.05, 0.08, 0.1]
    lamb_ii = [el * np.diag(XtX) for el in lambd]
    lamb = [np.diag(el) for el in lamb_ii]
    est_theta = [np.matmul(np.matmul(np.linalg.inv(XtX + el), matr_X.T), y) for el in lamb]
    norm = [(np.linalg.norm(el)) ** 2 for el in est_theta]
    theta = np.array([1.,1.,1.,1.,1.,1.,1.])
    norm_1 = (np.linalg.norm(theta - est_theta[4]))
    RSS = [np.matmul((y - np.matmul(matr_X, el)).T, (y - np.matmul(matr_X, el))) for el in est_theta]
    return est_theta, norm, RSS, norm_1, lambd
##################################
def Graph(x, y):
    plt.xlabel(r'$\lambda')
    p1 = plt.plot(x, y, 'r')
    plt.show()

########################
def estimation_PCA(matr_X, y, N):
    x_mean = np.mean(matr_X, axis=0)
    y_mean = np.mean(y)
    Y_s = np.array([el - y_mean for el in y])
    X_s = np.array([el - x_mean for el in matr_X])
    #X_s1 = X_s[:, 1:8]
    eig_XstXs, eig_vec = np.linalg.eig(np.matmul(X_s.T, X_s))
    ####
    eig_v = [el for el in eig_vec.T]
    arr = [[val, vec]  for val, vec in zip(eig_XstXs, eig_v)]
    eig = sorted(arr,  key=lambda eig_XstXs: eig_XstXs, reverse=True)
    eig_1 = [val for va in eig for val in va]
    ####
    eig_vals = eig_1[::2]
    s_eig = np.sum(eig_vals)
    eig_contrib = [el / s_eig for el in eig_vals]
    ####
    V = eig_1[1::2]
    ####
    eig_vals_1 = np.array(eig_vals[:6])
    V_1 = np.array(V[:6])
    #b = np.array(eig_vals[2:8])
    #bv = V[2:8]
    #eig_vector = np.vstack((a, b))
    #eig_val = np.hstack((av, bv))
    ######
    Z = np.matmul(X_s, V_1.T)
    ZtZ = np.matmul(Z.T, Z)
    #ZtZ = np.diag(eig_val) 
    b = np.matmul(np.matmul(np.linalg.inv(ZtZ), Z.T), Y_s) 
    est_theta = np.matmul(V_1.T, b)

    #################
    theta = np.array([1.,1.,1.,1.,1.,1.,1.])
    norm = (np.linalg.norm(theta - est_theta))
    RSS = np.matmul((y - np.matmul(matr_X, est_theta)).T, (y - np.matmul(matr_X, est_theta)))
    return est_theta, norm, RSS