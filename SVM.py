import numpy as np
import quadprog


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """ Solve a QP of the form min 1/2xTPx + qTx st Gx < h st Ax=b"""
    #qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_G = P
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def evaluate_alpha(alpha, K, label):
    """ Return success percent """
    result = K.dot(alpha)

    success = [float(result[i,0]*label[i] > 0) for i in range(len(label))]

    return np.mean(success)*100

def compute_G_h(label, lamb):
    nb_spl = len(label)
    G = np.zeros([nb_spl*2, nb_spl])
    h = np.zeros((nb_spl*2, ))
    for i in range(nb_spl):
        G[i,i] = - float(label[i])
        G[nb_spl + i, i] = float(label[i])
        h[nb_spl + i] = 1./(2.*lamb*float(nb_spl))
    return G, h

def svm_compute_label(data_in_kernel, alpha):
    """ Compute the label for the data given (in the form data[i,j] = K(x, xj) with x a new data, xj in the data set"""
    result = data_in_kernel.dot(alpha)
    return [int(result[i,0] > 0.) for i in range(data_in_kernel.shape[0])]

def SVM(K, label, lamb, K_test, label_test):
    G, h = compute_G_h(label, lamb)
    alpha = quadprog_solve_qp(P=K, q = - np.asarray(label).reshape((len(label),)), G = G, h = h)
    alpha = alpha.reshape([alpha.shape[0], 1])
    print("on train: ", evaluate_alpha(alpha, K, label))
    print("on test: ", evaluate_alpha(alpha, K_test, label_test) )
    return alpha


""" Just an example of how quadprog works :
M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)
q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.]).reshape((3,))
al = quadprog_solve_qp(P, q, G, h)
print(al)"""