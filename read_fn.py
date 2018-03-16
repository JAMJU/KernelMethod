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