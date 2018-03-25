import numpy as np
from lev_distance import levenshtein_dist
import itertools


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
    """ Return a list of nb of time encoding of each acid amin appears in each sequence"""
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
    """ Return representation acid + quad concatenanted"""
    acid_values = np.asarray(convert_acid_kernel(sequences, dico_acides))
    quad_values = np.asarray(convert_spectral_kernel_quad(sequences, list_seq_to_id))
    return np.concatenate((acid_values, quad_values), axis=1)

def convert_lect_trig(sequences, list_seq_id, beg = 0):
    """ Read the sequence 3 by 3 beginning at beg"""
    final = []
    for j in range(len(sequences)):
        sequence = sequences[j]
        dico_appear = {seq: 0 for seq in list_seq_id}
        for i in range(beg, len(sequence) - 2, 3):
            seq_to_add = sequence[i] + sequence[i+1] + sequence[i+2]
            dico_appear[seq_to_add] += 1
        final.append([dico_appear[k] for k in list_seq_id])
    return final

def convert_lect_acid(sequences, dico_acides, beg=0):
    """ Return a list of nb of time encoding of each acid amin appears in each sequence reading from beg 3 by 3"""
    final = []
    for j in range(len(sequences)):
        sequence = sequences[j].replace('T', 'U')
        dico_appear = {acid: 0 for acid in dico_acides.keys()}
        for i in range(beg,len(sequence) - 2, 3):
            seq_to_add = sequence[i] + sequence[i + 1] + sequence[i + 2]

            for acid in dico_acides.keys():
                if seq_to_add in dico_acides[acid]:
                    dico_appear[acid] += 1
        final.append([dico_appear[ac] for ac in dico_acides.keys()])
    return final

def get_mismatch_dict(list_seq_id):
    """ Return distances of levenstein btween each sequences in a dictionnary"""
    nb_mismatch = dict()
    for seq1 in list_seq_id:
        for seq2 in list_seq_id:
            if not seq1 + seq2 in nb_mismatch.keys():
                lev = levenshtein_dist(seq1, seq2)
                nb_mismatch[seq1 + seq2] = lev
                nb_mismatch[seq2 + seq1] = lev
    print("Mismatches possible computed")
    return nb_mismatch

def get_mismatches_from_one(sequence, list_let, nb_mis):
    """ Return all the sequences possible from sequence with mismatch = nb_mis"""
    nb_let = len(list_let)
    list_seq_final = []
    len_seq = len(sequence)
    nb = range(len_seq)
    list_mismatch_place = list(itertools.combinations(nb, nb_mis))
    list_letters_place = list(itertools.combinations(range(nb_let - 1), nb_mis))
    for i in range(len(list_mismatch_place)):
        list_seq = [sequence[0:list_mismatch_place[i][0]] for k in range(len(list_letters_place))]
        for j in range(len(list_mismatch_place[i])):
            letter = sequence[list_mismatch_place[i][j]]
            letters_to_use = [let  for let in list_let if let != letter]
            for m in range(len(list_seq)):
                list_seq[m] += letters_to_use[list_letters_place[m][j]]
                if j!= len(list_mismatch_place[i]) - 1:
                    list_seq[m] += sequence[list_mismatch_place[i][j] + 1:list_mismatch_place[i][j+1]]
                else:
                    list_seq[m] += sequence[list_mismatch_place[i][j] + 1:len_seq]
        list_seq_final += list_seq
    return list_seq_final


def get_correspondances(list_seq_id, nb_mismatch, list_let):
    """ Return dictionnary of sequances and their correspondance with mismatch"""
    len_seq = len(list_seq_id[0])
    dico_corres = dict()
    for seq in list_seq_id:
        already_in = False
        for seq2 in dico_corres.keys():
            if seq in dico_corres[seq2]:
                already_in = True
        if not already_in:
            dico_corres[seq] = [seq]
            for i in range(1, nb_mismatch + 1):
                dico_corres[seq] += get_mismatches_from_one(seq, list_let, i)
    return dico_corres

def get_full_corres(list_seq_id, nb_mismatch, list_let):
    """ Return dictionnary of all sequences and their correspondance with mismatch"""
    len_seq = len(list_seq_id[0])
    dico_corres = dict()
    for seq in list_seq_id:
        dico_corres[seq] = [seq]
        for i in range(1, nb_mismatch + 1):
            dico_corres[seq] += get_mismatches_from_one(seq, list_let, i)
    return dico_corres


def convert_mismatch_dico(sequences, dico_corres, list_corres,  list_seq_id):
    size_seq = len(list_seq_id[0])
    list_final = []
    for i in range(len(sequences)):
        vect_in = {seq:0 for seq in dico_corres.keys()}
        for j in range(len(sequences[i]) - size_seq + 1):
            seq_to_study = sequences[i][j: j+ size_seq + 1]
            for k in dico_corres.keys():
                if seq_to_study in dico_corres[k]:
                    vect_in[k] += 1.
        list_final.append([vect_in[t] for t in list_corres])
    return list_final


def convert_mismatch_lev(sequences,list_seq_id,  nb_mismatch, size_seqID, nb_mismatch_limit):
    """ Return kernel representation for list_seq_id up to nb_mismatch_limit nb of mismatches"""
    final = []

    for j in range(len(sequences)):
        sequence = sequences[j]
        dico_appear = {seq: 0 for seq in list_seq_id}
        for i in range(len(sequence) - size_seqID + 1):
            seq = sequence[i:i+size_seqID ]

            for s in list_seq_id:
                lev = nb_mismatch[seq + s]
                if lev <= nb_mismatch_limit:
                    dico_appear[s] += 1
        final.append([dico_appear[k] for k in list_seq_id])
    return final

# Computation

def gaussian_func(vect1, vect2, sigma):
    norm = np.linalg.norm(vect1 - vect2)
    gauss_val = np.exp(-(norm*norm)/(2*sigma*sigma))
    return gauss_val

def compute_K_gaussian(list_sequences_converted, sigma):
    """ Compute the gram matrix for the gaussian kernel"""
    K = [[[]for y in list_sequences_converted] for x in list_sequences_converted]
    for (i, phi_vect_i) in enumerate(list_sequences_converted):
        phi_vect_i = np.asarray(phi_vect_i, dtype = float).reshape(1,len(phi_vect_i))
        for (j, phi_vect_j) in enumerate(list_sequences_converted):
            phi_vect_j = np.asarray(phi_vect_j, dtype = float).reshape(1,len(phi_vect_j))
            K[i][j] = gaussian_func(phi_vect_i, phi_vect_j, sigma)
    return np.asarray(K).reshape(len(list_sequences_converted), len(list_sequences_converted))

def compute_K_matrix(list_sequences_converted):
    """ Compute the gram matrix"""
    list_seq = np.asarray(list_sequences_converted, dtype = float)
    return list_seq.dot(list_seq.T)

def compute_test_matrix(training, testing):
    """ Compute the gram matrix for the test set"""
    return testing.dot(training.T)
