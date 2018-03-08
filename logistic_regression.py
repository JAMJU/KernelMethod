import numpy as np
# parameter to avoid the sigmoid function to return 0 because of an overflow of the exp function
epsilon = 10**-10

def sigmoid(u):
    if u < 0.:
        return  max(epsilon, np.asscalar(np.exp(u))/( np.asscalar(np.exp( u)) + 1.))
    else:

        return max(epsilon, 1./(1. + np.asscalar(np.exp(- u))))

def solveWKRR(K, W, z, lamb):
    """ Solve the WKRR found for kernel logistic regression"""
    first = np.linalg.inv((W**(1./2.)).dot(K).dot((W**(1./2.))) + float(K.shape[0])*lamb*np.identity(K.shape[0]))
    return (W**(1./2.)).dot(first).dot((W**(1./2.))).dot(z)

def evaluate_alpha(alpha, K, label):
    """ Return success percent """
    result = K.dot(alpha)

    success = [float(result[i,0]*label[i] > 0) for i in range(len(label))]

    return np.mean(success)*100

def compute_loss(K, alpha, lamb, label):
    lab = np.asarray(label).reshape([len(label), 1])
    result = K.dot(alpha)
    loss = np.log(1 + np.exp(np.multiply(-result, lab)))
    loss = (1./float(len(label)))*np.sum(loss) + lamb*alpha.T.dot(K).dot(alpha)
    return loss


def logistic_kernel_regression(K, label, lamb, nb_it):
    """ Apply logistic kernel regression with solving by newton method
    with K the kernel matrix and alpha0 the initialization
    return the alpha found """
    # Initialisation
    alpha = (1. - 2.*np.random.random_sample([K.shape[0] , 1]))

    p = np.zeros([K.shape[0], 1])
    w = np.zeros([K.shape[0], K.shape[0]])
    z = np.zeros([K.shape[0], 1])
    for j in range(nb_it):
        m = K.dot(alpha)

        for i in range(K.shape[0]):
            p[i, 0] = - sigmoid(-label[i] * m[i, 0])
            w[i, i] = sigmoid(m[i, 0]) * sigmoid(-m[i, 0])
            z[i, 0] = m[i, 0] - label[i] * p[i,0]/ (w[i,i])
        alpha = solveWKRR(K, w, z, lamb)
        print "loss", compute_loss(K, alpha, lamb, label)
        print "iteration ", j, evaluate_alpha(alpha, K, label)

    return alpha

def compute_label(data_in_kernel, alpha):
    """ Compute the label for the data given (in the form data[i,j] = K(x, xj) with x a new data, xj in the data set"""
    result = data_in_kernel.dot(alpha)
    return [int(result[i,0] > 0.) for i in range(data_in_kernel.shape[0])]







