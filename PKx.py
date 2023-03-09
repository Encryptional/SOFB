import os
import numpy as np
import sys

def BuildFeatureDictionary():
    Feature_table = np.array([7.00, 7.00, 3.65, 3.22, 7.00, 7.00, 6.00, 7.00, 10.53, 7.00,
                              7.00, 8.18, 7.00, 7.00, 12.48, 7.00, 7.00, 7.00, 7.00, 10.07, 7.0])
    max_Feature = np.amax(Feature_table)
    min_Feature = np.amin(Feature_table)

    normolized_Feature_table = (Feature_table - min_Feature) / (max_Feature - min_Feature)


    Feature_dict = {}
    Feature_dict['A'] = normolized_Feature_table[0]
    Feature_dict['C'] = normolized_Feature_table[1]
    Feature_dict['D'] = normolized_Feature_table[2]
    Feature_dict['E'] = normolized_Feature_table[3]
    Feature_dict['F'] = normolized_Feature_table[4]
    Feature_dict['G'] = normolized_Feature_table[5]
    Feature_dict['H'] = normolized_Feature_table[6]
    Feature_dict['I'] = normolized_Feature_table[7]
    Feature_dict['K'] = normolized_Feature_table[8]
    Feature_dict['L'] = normolized_Feature_table[9]
    Feature_dict['M'] = normolized_Feature_table[10]
    Feature_dict['N'] = normolized_Feature_table[11]
    Feature_dict['P'] = normolized_Feature_table[12]
    Feature_dict['Q'] = normolized_Feature_table[13]
    Feature_dict['R'] = normolized_Feature_table[14]
    Feature_dict['S'] = normolized_Feature_table[15]
    Feature_dict['T'] = normolized_Feature_table[16]
    Feature_dict['V'] = normolized_Feature_table[17]
    Feature_dict['W'] = normolized_Feature_table[18]
    Feature_dict['Y'] = normolized_Feature_table[19]
    Feature_dict['X'] = normolized_Feature_table[20]

    return Feature_dict



def GetFeature(AA, Feature_dict):
    if (AA not in Feature_dict):
        print("[warning]: Feature_dict can't find ", AA, ". Returning 0")
        return 0
    else:
        return Feature_dict[AA]

def RetriveFeatureFromASequence(seq, Feature_dict):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 2)
    Feature = []
    for index, item in enumerate(seq):
        Feature.append(GetFeature(item, Feature_dict))
    return Feature


def load_fasta_and_compute(seq_fn, out_fn, Feature_dict):
    fin = open(seq_fn, "r")
    f = fin.readlines()
    fout = open(out_fn, "w")
    for i in range(0, len(f), 3): ##set =4 when it is train set and =3 when it is test set
        line_PID = f[i].rstrip("\n")
        line_PID = f[i].rstrip("\n")
        line_Seq = f[i + 1].rstrip("\n")
        fout.write(line_PID + "\n")
        fout.write(line_Seq + "\n")
        Feature = RetriveFeatureFromASequence(line_Seq, Feature_dict)
        fout.write(" ".join(map(str,Feature)) + "\n")
    fin.close()
    fout.close()

if __name__ == '__main__':
    Feature_dict = BuildFeatureDictionary()
    seq_fn = './Datasets/RNA/RNA-117_Test.txt'
    out_fn = './multi-feature/RNA/RNA_test_PKx.txt'
    load_fasta_and_compute(seq_fn, out_fn, Feature_dict)
    fil = open(out_fn, "r")
    g = fil.readlines()
    PKx= []
    for i in range(0, (len(g)), 3):
        line = g[i+2].split(' ')
        PKx.append(line)
    # print(RSA[0][0])
    pkx_value=[]
    for i in range(len(PKx)):
        for j in range(len(PKx[i])):
            pkx_value.append(PKx[i][j])
    pkx = np.array(pkx_value)
    np.save('./multi-feature/RNA/pkx_test.npy', pkx)

