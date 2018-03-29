import numpy as np
from logistic_regression import logistic_kernel_regression, compute_label
from kernel_creation import convert_spectral_kernel_quad, convert_spectral_kernel_quint, convert_spectral_kernel_trig
from kernel_creation import convert_acid_kernel, convert_acid_quad, convert_mismatch_lev, convert_lect_trig, get_mismatch_dict
from kernel_creation import get_correspondances, convert_mismatch_dico, get_full_corres, convert_encode
from kernel_creation import compute_test_matrix, compute_K_matrix, convert_lect_acid, compute_K_gaussian
from read_fn import read_csv_file_label, read_csv_file_data, save_label, save_data_converted
from SVM import SVM, svm_compute_label

list_letters = ["A", "C", "G", "T"]
list_trig = [a + b + c for a in list_letters for b in list_letters for c in list_letters]
list_quad =  [a + b + c + d for a in list_letters for b in list_letters for c in list_letters for d in list_letters]
list_quint = [a + b + c + d  + e for a in list_letters for b in list_letters for c in list_letters for d in list_letters for e in list_letters]
list_six = [a + b + c + d  + e + f for a in list_letters for b in list_letters for c in list_letters for d in list_letters for e in list_letters for f in list_letters]
dico_acid = {'Alanine': [ 	'GCU', 'GCC', 'GCA', 'GCG'], 'Arginine': ['CGU', 'CGC', 'CGA', 'CGG' , 'AGA', 'AGG'],
             'Asparagine': ['AAU', 'AAC'], 'Acide aspartique': ['GAU', 'GAC'],
             'Cysteine': ['UGU', 'UGC'], 'Glutamine': ['CAA', 'CAG'], 'Acide glutamique':['GAA', 'GAG'],
             'Glycine':['GGU', 'GGC', 'GGA', 'GGG'], 'Histidine': ['CAU', 'CAC'], 'Isoleucine': ['AUU', 'AUC', 'AUA'],
             'Leucine': ['UUA', 'UUG' , 'CUU', 'CUC', 'CUA', 'CUG'], 'Lysine': ['AAA', 'AAG'],
             'Methionine': ['AUG'], 'Phenylalanine':['UUU', 'UUC'], 'Proline' :['CCU', 'CCC', 'CCA', 'CCG'],
             'Pyrrolysine': ['UAG'], 'Selenocysteine':['UGA'], 'Serine':['UCU', 'UCC', 'UCA', 'UCG' , 'AGU', 'AGC'],
             'Threonine':['ACU', 'ACC', 'ACA', 'ACG'], 'Tryptophane':['UGG'], 'Tyrosine':['UAU', 'UAC'],
             'Valine':['GUU', 'GUC', 'GUA', 'GUG'], 'Initiation': ['AUG'], 'Terminaison': ['UAG', 'UAA', 'UGA']}


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

## Parameters
lamb_log = 0.0000001
lamb_svm = 0.00001
sigma = 0.8
add_param = 10.**(-10)
list_seq_id = list_six
mis_lev = False
if mis_lev:
    dict_mismatch = get_mismatch_dict(list_seq_id)


mis_dic = False
size_seq = 6
nb_mis = 0
beg = 0
if mis_dic:
    dict_corres = get_correspondances(list_seq_id, nb_mis, list_letters)
    list_mis_corres = dict_corres.keys()
    print(list_mis_corres)
mis_dic_full = False
if mis_dic_full:
    dict_corres = get_full_corres(list_seq_id, nb_mis, list_letters)
    list_mis_corres = dict_corres.keys()

##
list_labels_log = []
list_labels_svm = []
for name in [ "0", "1","2"]:
    print ("beginning loading of the data")
    # Training data
    sequences = read_csv_file_data("data/Xtr"+ name+ ".csv")
    #list_converted = convert_spectral_kernel_trig(sequences, list_seq_id)
    #list_converted = convert_spectral_kernel_quad(sequences, list_quad)
    list_converted = convert_spectral_kernel_quint(sequences, list_quint)
    #list_converted = convert_spectral_kernel_quint(sequences, list_quint)
    #list_converted = convert_acid_kernel(sequences, dico_acid)
    #list_converted = convert_acid_quad(sequences, dico_acid, list_quad

    #list_converted = convert_mismatch_lev(sequences,  list_seq_id, dict_mismatch,  size_seq, nb_mis)
    #list_converted = convert_lect_trig(sequences, list_seq_id, beg)
    #list_converted = convert_lect_acid(sequences, dico_acid, beg)
    #list_converted = convert_mismatch_dico(sequences, dict_corres,list_mis_corres, list_seq_id)
    #list_converted = convert_encode(sequences, list_letters)
    training = np.asarray(list_converted, dtype = float)

    # to avoid huge values and to save time for the logistic regression :
    sm = np.sum(training, axis= 1)
    training = training/sm[0]
    mean =  np.mean(training, axis= 0)

    training = training - mean

    #vst = np.std(training, axis= 0)
    #training = training / vst
    #save_data_converted("spectral_kernel/Xtr"+ name+ ".csv", training)

    # label training data
    label = read_csv_file_label("data/Ytr"+ name+ ".csv")
    label= np.asarray(label).reshape((len(label), ))

    # select what will be the test for training
    size_test = int(training.shape[0]/10)
    test_train = training[0:size_test]
    label_test_train = label[0:size_test]
    print( label_test_train.shape)
    size_total = training.shape[0]
    training = training[size_test:size_total]
    label_train = label[size_test:size_total]
    print (label_train.shape)

    # Test data
    sequences_test = read_csv_file_data("data/Xte"+ name+ ".csv")
    #list_converted_test = convert_spectral_kernel_trig(sequences_test, list_seq_id)
    #list_converted_test = convert_spectral_kernel_quad(sequences_test, list_quad)
    list_converted_test = convert_spectral_kernel_quint(sequences_test, list_quint)
    #list_converted_test = convert_acid_kernel(sequences_test, dico_acid)
    #list_converted_test = convert_acid_quad(sequences_test, dico_acid, list_quad)
    #list_converted_test = convert_mismatch_lev(sequences_test, list_seq_id, dict_mismatch, size_seq, nb_mis)
    #list_converted_test = convert_lect_trig(sequences_test, list_seq_id, beg )
    #list_converted_test = convert_lect_acid(sequences_test, dico_acid, beg)
    #list_converted_test = convert_mismatch_dico(sequences_test, dict_corres,list_mis_corres, list_seq_id)
    #list_converted_test = convert_encode(sequences, list_letters)
    testing = np.asarray(list_converted_test, dtype = float)
    # to avoid huge values and to save time for the logistic regression :
    testing = testing/sm[0]
    testing = testing - mean

    #testing = testing/ vst
    # param for each dataset:
    """if name=="0":
        lamb_svm = 0.000008
        add_param = 10. ** (-10)

    if name=="1":
        lamb_svm = 0.00001
        add_param = 10.**(-10)

    if name == "2":
        lamb_svm = 0.000005
        add_param=10.**(-9)"""

    if name=="2":
        add_param = 10**(-9)


    print ("data loaded")

    # Computing the kernel
    print ("beginning computing K")
    K = compute_K_matrix(training)
    add = add_param*np.identity(K.shape[0])
    K_add = K + add # to make it positive definite
    #K = compute_K_gaussian(training, sigma)
    #K_add = K
    print(K)
    print("K shape", K.shape)
    print(is_pos_def(K_add))
    K_test_train = compute_test_matrix(training, test_train)
    print (K_test_train.shape)
    print ("K computed")

    """#Training : kernel logistic regression
    alpha = logistic_kernel_regression(K, label_train, lamb_log, 15, K_test_train, label_test_train)


    # Testing : kernel logistic regression
    Ktest = compute_test_matrix(training, testing)
    labels_test = compute_label(Ktest, alpha)
    list_labels_log = list_labels_log + labels_test"""

    # Training : SVM
    alpha = SVM(K_add, label_train, lamb_svm, K_test_train, label_test_train)
    print(alpha)
    # Testing : kernel logistic regression
    Ktest = compute_test_matrix(training, testing)
    labels_test = svm_compute_label(Ktest, alpha)
    list_labels_svm = list_labels_svm + labels_test


save_label(0, list_labels_svm,"results/SVM-quint-centered-mixed.csv" )
