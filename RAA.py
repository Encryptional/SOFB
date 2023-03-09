import numpy as np
import sys

def BuildRAADictionary():
    RAA_table = np.array(
        [-0.08, 0.12, -0.15, -0.33, 0.76, -0.11, -0.34, -0.25, 0.18, 0.71,
         0.61, -0.38, 0.92, 1.18, -0.17, -0.13, -0.07, 0.95, 0.71, 0.37])
    max_RAA = np.amax(RAA_table)
    min_RAA = np.amin(RAA_table)

    normolized_RAA_table = (RAA_table - min_RAA) / (max_RAA - min_RAA)

    RAA_dict = {}
    RAA_dict['A'] = normolized_RAA_table[0]
    RAA_dict['R'] = normolized_RAA_table[1]
    RAA_dict['N'] = normolized_RAA_table[2]
    RAA_dict['D'] = normolized_RAA_table[3]
    RAA_dict['C'] = normolized_RAA_table[4]
    RAA_dict['Q'] = normolized_RAA_table[5]
    RAA_dict['E'] = normolized_RAA_table[6]
    RAA_dict['G'] = normolized_RAA_table[7]
    RAA_dict['H'] = normolized_RAA_table[8]
    RAA_dict['I'] = normolized_RAA_table[9]
    RAA_dict['L'] = normolized_RAA_table[10]
    RAA_dict['K'] = normolized_RAA_table[11]
    RAA_dict['M'] = normolized_RAA_table[12]
    RAA_dict['F'] = normolized_RAA_table[13]
    RAA_dict['P'] = normolized_RAA_table[14]
    RAA_dict['S'] = normolized_RAA_table[15]
    RAA_dict['T'] = normolized_RAA_table[16]
    RAA_dict['W'] = normolized_RAA_table[17]
    RAA_dict['Y'] = normolized_RAA_table[18]
    RAA_dict['V'] = normolized_RAA_table[19]

    return RAA_dict

def GetRAA(AA, RAA_dict):
    if (AA not in RAA_dict):
        print("[warning]: RAA_dict can't find ", AA, ". Returning 0")
        return 0
    else:
        return RAA_dict[AA]

def RetriveRAAFromASequence(seq, RAA_dict):
    seq = seq.rstrip('\n').rstrip(' ')
    assert (len(seq) >= 2)
    raa = []
    for index, item in enumerate(seq):
        raa.append(GetRAA(item, RAA_dict))
    return raa

def load_fasta_and_compute(seq_fn, out_fn, RAA_dict):
    fin = open(seq_fn, "r")
    f = fin.readlines()
    fout = open(out_fn, "w")
    for i in range(0, len(f), 3): ##set =4 when it is train set and =3 when it is test set
        line_PID = f[i].rstrip("\n")
        line_Seq = f[i+1].rstrip("\n")
        fout.write(line_PID + "\n")
        fout.write(line_Seq + "\n")
        raa = RetriveRAAFromASequence(line_Seq, RAA_dict)
        fout.write(" ".join(map(str,raa)) + "\n")
    fin.close()
    fout.close()

if __name__ == '__main__':
    RAA_dict = BuildRAADictionary()
    seq_fn = './Datasets/RNA/RNA-117_Test.txt' ##datasets_path
    out_fn = './multi-feature/RNA/test_RNA_RAA.txt' ##result_path
    load_fasta_and_compute(seq_fn, out_fn, RAA_dict)
    fil = open(out_fn, "r")
    g = fil.readlines()
    RAA = []
    for i in range(0, (len(g)), 3):
        line = g[i + 2].split(' ')
        RAA.append(line)
    raa_value = []
    for i in range(len(RAA)):
        for j in range(len(RAA[i])):
            raa_value.append(RAA[i][j])
    raa = np.array(raa_value)
    np.save('./multi-feature/RNA/raa_train.npy', raa)
