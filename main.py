import numpy as np
from logistic_regression import logistic_kernel_regression, compute_label
list_letters = ["A", "C", "G", "T"]
list_trig = [a + b + c for a in list_letters for b in list_letters for c in list_letters]


def read_csv_file_data(namefile):
    """ Read a csv file that is supposed to containe DNA sequences"""
    sequences = []
    with open(namefile, 'r') as f:
        for line in f:
            lin = line.replace('\n', '')
            sequences.append(lin)
    return sequences

def read_csv_file_label(namefile):
    """ Read csv file that is supposed to be with two columns, one the index, two the label"""
    label = []
    with open(namefile, "r") as f:
        for line in f:
            lin = line.replace('\n' ,'')
            list_lab = lin.split(",")
            if list_lab[0] != "":
                label.append(-1. if float(list_lab[1]) == 0. else 1.)
    return label

def convert_spectral_kernel(sequences, list_seq_to_id):
    """ Return a list seq of nb of time the seq in list_seq_to_id appear in sequence"""
    final = []
    for j in range(len(sequences)):
        sequence = sequences[j]
        dico_appear = {seq: 0 for seq in list_seq_to_id}
        for i in range(len(sequence) - 2):
            seq_to_add = sequence[i] + sequence[i+1] + sequence[i+2]
            dico_appear[seq_to_add] += 1

        final.append([dico_appear[k] for k in list_seq_to_id])
    return final

def save_data_converted(namefile, dbl_seq):
    with open(namefile, 'w') as f:
        for i in range(len(dbl_seq)):
            first = True
            for k in dbl_seq[i]:
                if not first:
                    f.write(",")
                else:
                    first = False
                f.write(str(k))
            f.write("\n")

def save_label(begin_index, labels, namefile):
    with open(namefile, 'w') as f:
        f.write("Id,Bound\n")
        for i in range(len(labels)):
            f.write(str(begin_index + i) + "," + str(labels[i]))
            f.write("\n")

def compute_K_matrix(list_sequences_converted):
    """ Compute the gram matrix"""
    width = len(list_sequences_converted[0])
    K = np.zeros([len(list_sequences_converted), len(list_sequences_converted)])
    for i in range(len(list_sequences_converted)):
        for j in range(len(list_sequences_converted)):
            x1 = np.asarray(list_sequences_converted[i], dtype= float).reshape(width, 1)
            x2 = np.asarray(list_sequences_converted[j], dtype = float).reshape(width, 1)
            K[i,j] = float(np.asscalar(x1.T.dot(x2)))

    return K

def compute_test_matrix(training, testing):
    """ Compute the gram matrix for the test set"""
    return testing.dot(training.T)

#################################################### LOGISTIC ##############################################
## Parameters of the logistic regression
lamb = 0.000001
##
list_labels = []
for name in ["0", "1", "2"]:
    print "beginning loading of the data"
    # Training data
    sequences = read_csv_file_data("data/Xtr"+ name+ ".csv")
    list_converted = convert_spectral_kernel(sequences, list_trig)
    training = np.asarray(list_converted, dtype = float)
    # to avoid huge values and to save time for the logistic regression :
    sm =  np.sum(training, axis= 1)
    training = training/sm[0]
    save_data_converted("spectral_kernel/Xtr"+ name+ ".csv", training)

    # label training data
    label = read_csv_file_label("data/Ytr"+ name+ ".csv")
    label= np.asarray(label).reshape((len(label), ))

    # Test data
    sequences_test = read_csv_file_data("data/Xte"+ name+ ".csv")
    list_converted_test = convert_spectral_kernel(sequences_test, list_trig)
    testing = np.asarray(list_converted_test, dtype = float)
    # to avoid huge values and to save time for the logistic regression :
    testing = testing/sm[0]

    print "data loaded"

    # Training : kernel logistic regression
    print "beginning computing K"
    K = compute_K_matrix(training)
    print "K computed"

    alpha = logistic_kernel_regression(K,label, lamb, 10)


    # Testing : kernel logistic regression
    Ktest = compute_test_matrix(training, testing)
    labels_test = compute_label(Ktest, alpha)
    list_labels = list_labels + labels_test

save_label(0, list_labels,"results/LKR_0000001.csv" )











