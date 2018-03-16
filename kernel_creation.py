import numpy as np



def convert_spectral_kernel_trig(sequences, list_seq_to_id):
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

def convert_spectral_kernel_quad(sequences, list_seq_to_id):
    """ Return a list seq of nb of time the seq in list_seq_to_id appear in sequence"""
    final = []
    for j in range(len(sequences)):
        sequence = sequences[j]
        dico_appear = {seq: 0 for seq in list_seq_to_id}
        for i in range(len(sequence) - 3):
            seq_to_add = sequence[i] + sequence[i+1] + sequence[i+2] + sequence[i+3]
            dico_appear[seq_to_add] += 1
        final.append([dico_appear[k] for k in list_seq_to_id])
    return final

def convert_spectral_kernel_quint(sequences, list_seq_to_id):
    """ Return a list seq of nb of time the seq in list_seq_to_id appear in sequence"""
    final = []
    for j in range(len(sequences)):
        sequence = sequences[j]
        dico_appear = {seq: 0 for seq in list_seq_to_id}
        for i in range(len(sequence) - 4):
            seq_to_add = sequence[i] + sequence[i+1] + sequence[i+2] + sequence[i+3] + sequence[i+4]
            dico_appear[seq_to_add] += 1
        final.append([dico_appear[k] for k in list_seq_to_id])
    return final

def convert_acid_kernel(sequences, dico_acides):
    final = []
    for j in range(len(sequences)):
        sequence = sequences[j].replace('T', 'U')
        dico_appear = {acid: 0 for acid in dico_acides.keys()}
        for i in range(len(sequence) - 2):
            seq_to_add = sequence[i] + sequence[i + 1] + sequence[i + 2]

            for acid in dico_acides.keys():
                if seq_to_add in dico_acides[acid]:

                    dico_appear[acid] += 1
        final.append([dico_appear[ac] for ac in dico_acides.keys()])
    return final


def convert_acid_quad(sequences, dico_acides, list_seq_to_id):
    acid_values = np.asarray(convert_acid_kernel(sequences, dico_acides))
    quad_values = np.asarray(convert_spectral_kernel_quad(sequences, list_seq_to_id))
    return np.concatenate((acid_values, quad_values), axis=1)

def compute_K_matrix(list_sequences_converted):
    """ Compute the gram matrix"""

    list_seq = np.asarray(list_sequences_converted, dtype = float)
    return list_seq.dot(list_seq.T)

def compute_test_matrix(training, testing):
    """ Compute the gram matrix for the test set"""
    return testing.dot(training.T)