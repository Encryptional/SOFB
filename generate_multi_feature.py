import random
import pickle
import numpy as np
import math
import os
import argparse
import torch
from transformers import BertModel, BertTokenizer
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--nucleic_acid", dest="nucleic_acid", help="Nucleic_Acid type. It can be chosen from DNA,RNA.")
    return parser.parse_args()

def read_fasta(train_dir, test_dir):
    seqdict = {}
    train_list = []
    train_seq = []
    test_list = []
    test_seq = []
    train_label = []
    test_label = []
    with open(train_dir, 'r') as f:
        train_text = f.readlines()
    for i in range(0, len(train_text), 4):
        id = train_text[i].strip()[1:]
        if id[-1].islower():
            id += id[-1]
        pro_seq = train_text[i + 1].strip()
        pro_anno = train_text[i + 2].strip()
        train_list.append(id)
        seqdict[id] = {'seq': pro_seq, 'anno': pro_anno}
    for name in train_list:
        train_seq.append(seqdict[name]['seq'])
        train_label.append(seqdict[name]['anno'])

    with open(test_dir, 'r') as f:
        test_text = f.readlines()
    for i in range(0, len(test_text), 3):
        id = test_text[i].strip()[1:]
        if id[-1].islower():
            id += id[-1]
        pro_seq = test_text[i + 1].strip()
        pro_ann = test_text[i + 2].strip()
        test_list.append(id)
        seqdict[id] = {'seq': pro_seq, 'anno': pro_ann}
    for name in test_list:
        test_seq.append(seqdict[name]['seq'])
        test_label.append(seqdict[name]['anno'])
    return train_list, test_list, train_seq, test_seq, train_label, test_label

def get_HMM(ligand,seq_list,hmm_dir,feature_dir):
    hmm_dict = {}
    for seqid in seq_list:
        file = seqid+'.hhm'
        with open(hmm_dir+'/'+file,'r') as fin:
            fin_data = fin.readlines()
            hhm_begin_line = 0
            hhm_end_line = 0
            for i in range(len(fin_data)):
                if '#' in fin_data[i]:
                    hhm_begin_line = i+5
                elif '//' in fin_data[i]:
                    hhm_end_line = i
            feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
            axis_x = 0
            for i in range(hhm_begin_line,hhm_end_line,3):
                line1 = fin_data[i].split()[2:-1]
                line2 = fin_data[i+1].split()
                axis_y = 0
                for j in line1:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                for j in line2:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                axis_x+=1
            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            hmm_dict[file.split('.')[0]] = feature
    with open(feature_dir + '/{}_HMM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(hmm_dict, f)
    return

def get_PSSM(ligand,seq_list,pssm_dir,feature_dir):
    nor_pssm_dict = {}
    for seqid in seq_list:
        file = seqid+'.pssm'
        with open(pssm_dir+'/'+file,'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for i in range(1,len(fin_data)):
                if fin_data[i] == '\n':
                    pssm_end_line = i
                    break
            feature = np.zeros([(pssm_end_line-pssm_begin_line),20])
            axis_x = 0
            for i in range(pssm_begin_line,pssm_end_line):
                raw_pssm = fin_data[i].split()[2:22]
                axis_y = 0
                for j in raw_pssm:
                    feature[axis_x][axis_y]= (1 / (1 + math.exp(-float(j))))
                    axis_y+=1
                axis_x+=1
            nor_pssm_dict[file.split('.')[0]] = feature
    with open(feature_dir+'/{}_PSSM.pkl'.format(ligand),'wb') as f:
        pickle.dump(nor_pssm_dict,f)
    return

def generate_label(train_label, test_label, save_dir):
    ##标签生成
    label_train = []
    label_test = []
    for i in range(len(train_label)):
        for j in range(len(train_label[i])):
            label_train.append(train_label[i][j])
    for i in range(len(test_label)):
        for j in range(len(test_label[i])):
            label_test.append(test_label[i][j])

    label_train = np.array(label_train)
    label_test = np.array(label_test)
    np.save(save_dir + '/' +'train_label', label_train)
    np.save(save_dir + '/' +'test_label', label_test)

def generate_PSSM_HMM(train_list, test_list, ligand):
    ##PSSM|HMM
    seqlist = train_list + test_list
    PSSM_dir = 'pssm_file_path'
    HMM_dir = 'hmm_file_path'
    if not((os.path.exists('./multi-feature/' + '{}'.format(ligand) + '/HMM.pkl')) & (os.path.exists('./multi-feature/' + '{}'.format(ligand) + '/PSSM.pkl'))):
        get_PSSM(ligand, seqlist, PSSM_dir, Dataset_dir)
        get_HMM(ligand, seqlist, HMM_dir, Dataset_dir)

    ##存储bio_feature至local（）
    feature_list = ['PSSM', 'HMM']
    for feature in feature_list:
        with open('./multi-feature/' + '{}'.format(ligand) + '/' + '{}.pkl'.format(feature), 'rb') as f:
            locals()[ligand + '_' + feature] = pickle.load(f)

    bio_PSSM = []
    bio_HMM = []
    for seq_id in train_list:
        pss = locals()[ligand + '_' + feature_list[0]][seq_id]
        hmm = locals()[ligand + '_' + feature_list[1]][seq_id]
        bio_PSSM.append(pss)
        bio_HMM.append(hmm)
    t_bio = []
    for i in range(len(bio_PSSM)):
        for j in range(bio_PSSM[i].shape[0]):
            t_bio.append(bio_PSSM[i][j][:])
    train_pss = np.array(t_bio)
    t_hmm = []
    for i in range(len(bio_HMM)):
        for j in range(bio_HMM[i].shape[0]):
            t_hmm.append(bio_HMM[i][j][:])
    train_hmm = np.array(t_hmm)
    ##test
    bio_fea = []
    bio_feast = []
    for seq_id in test_list:
        pss_test = locals()[ligand+'_' + feature_list[0]][seq_id]
        hmm_test = locals()[ligand+'_' + feature_list[1]][seq_id]
        bio_fea.append(pss_test)
        bio_feast.append(hmm_test)
    te_bio = []
    for i in range(len(bio_fea)):
        for j in range(bio_fea[i].shape[0]):
            te_bio.append(bio_fea[i][j][:])
    test_bio_pssm = np.array(te_bio)
    te_bios = []
    for i in range(len(bio_feast)):
        for j in range(bio_feast[i].shape[0]):
            te_bios.append(bio_feast[i][j][:])
    test_bio_hmm = np.array(te_bios)
    #save
    np.save('./multi-feature/' + '{}'.format(ligand) + '/' + 'train_bio_PSSM.npy', train_pss)
    np.save('./multi-feature/' + '{}'.format(ligand) + '/' + 'train_bio__HMM.npy', train_hmm)
    np.save('./multi-feature/' + '{}'.format(ligand) + '/' + 'test_bio_PSSM.npy', test_bio_pssm)
    np.save('./multi-feature/' + '{}'.format(ligand) + '/' + 'test_bio_HMM.npy', test_bio_hmm)

def get_dynamic_embdedding(sequences):

    tokenizer = BertTokenizer.from_pretrained("/home/zhangbin/SOFB-master/save_model/prot_bfd_ft8282", do_lower_case=False)
    model = BertModel.from_pretrained("/home/zhangbin/SOFB-master/save_model/prot_bfd_ft8282")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_split = []
    for i in sequences:
        seq_split.append(str(' '.join([word for word in i])))
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_split]
    features = []
    count = 0
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=8, shuffle=False)
    for i in dataloader:
        ids = tokenizer.batch_encode_plus(i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            count += 1
            features.append(seq_emd)
    vec_train = []
    for i in range(len(features)):
        for j in range(features[i].shape[0]):
            vec_train.append(features[i][j][:])
    vec_train = np.array(vec_train)
    return vec_train

def get_generalization_embedding(seq_list):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()
    #device = torch.device('cpu')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    #添加空格分隔
    seq_split = []
    for i in seq_list:
        seq_split.append(str(' '.join([word for word in i])))
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_split]
    features = []
    count = 0
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=8, shuffle=False)
    for i in dataloader:
        ids = tokenizer.batch_encode_plus(i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            count+=1
            features.append(seq_emd)
    vec_train = []
    for i in range(len(features)):
        for j in range(features[i].shape[0]):
            vec_train.append(features[i][j][:])
    vec_train = np.array(vec_train)
    return vec_train

def generate_one_hot(train_seq, test_seq):
    train_acid = []
    test_acid = []
    for i in range(len(train_seq)):
        for j in range(len(train_seq[i])):
            train_acid.append(train_seq[i][j])
    for i in range(len(test_seq)):
        for j in range(len(test_seq[i])):
            test_acid.append(test_seq[i][j])
    seq_acid_dict = {'G': 1, 'A': 2, 'V': 3, 'L': 4, 'I': 5, 'P': 6, 'F': 7, 'Y': 8, 'W': 9, 'S': 10,
                     'T': 11, 'C': 12, 'M': 13, 'N': 14, 'Q': 15, 'D': 16, 'E': 17, 'K': 18, 'R': 19, 'H': 0}
    train_acid_num = []
    test_acid_num = []
    for acid in train_acid:
        train_acid_num.append(seq_acid_dict[str(acid)])
    for acid in test_acid:
        test_acid_num.append(seq_acid_dict[str(acid)])

    train_ac_one = []
    for i in train_acid_num:
        nu = np.zeros(20)
        nu[i] = 1
        train_ac_one.append(nu)
    train_one_hot = np.array(train_ac_one)
    test_ac_one = []
    for i in test_acid_num:
        nu = np.zeros(20)
        nu[i] = 1
        test_ac_one.append(nu)
    test_one_hot = np.array(test_ac_one)
    np.save('./multi-feature/' + ligand + '/' + 'train_one_hot.npy', train_one_hot)
    np.save('./multi-feature/' + ligand + '/' +'test_one_hot.npy', test_one_hot)

def generate_bio(ligand):
    train_raa = (np.load('./multi-feature/' + ligand + '/raa_train.npy').reshape(-1,1)).astype(np.float)
    train_pca1 = (np.load('./multi-feature/' + ligand + '/pychar_train1.npy').reshape(-1,1)).astype(np.float)
    train_pca2 = (np.load('./multi-feature/' + ligand + '/pychar_train2.npy').reshape(-1,1)).astype(np.float)
    train_pca3 = (np.load('./multi-feature/' + ligand + '/pychar_train3.npy').reshape(-1,1)).astype(np.float)
    train_pkx = (np.load('./multi-feature/' + ligand + '/pkx_train.npy').reshape(-1,1)).astype(np.float)
    train_onehot = (np.load('./multi-feature/' + ligand + '/train_one_hot.npy')).astype(np.float)
    train_pssm = np.load('./multi-feature/' + ligand + '/train_bio_PSSM.npy')
    train_hmm = np.load('./multi-feature/' + ligand + '/train_bio__HMM.npy')


    test_raa = (np.load('./multi-feature/' + ligand + '/raa_test.npy').reshape(-1,1)).astype(np.float)
    test_pca1 = (np.load('./multi-feature/' + ligand + '/pychar_test1.npy').reshape(-1,1)).astype(np.float)
    test_pca2 = (np.load('./multi-feature/' + ligand + '/pychar_test2.npy').reshape(-1,1)).astype(np.float)
    test_pca3 = (np.load('./multi-feature/' + ligand + '/pychar_test3.npy').reshape(-1,1)).astype(np.float)
    test_pkx = (np.load('./multi-feature/' + ligand + '/pkx_test.npy').reshape(-1,1)).astype(np.float)
    test_onehot = (np.load('./multi-feature/' + ligand + '/test_one_hot.npy')).astype(np.float)
    test_pssm = np.load('./multi-feature/' + ligand + '/test_bio_PSSM.npy')
    test_hmm = np.load('./multi-feature/' + ligand + '/test_bio_HMM.npy')


    train_bio = np.concatenate((train_raa, train_pca1, train_pca2, train_pca3, train_pkx, train_pssm, train_hmm,train_onehot), axis=1)
    test_bio = np.concatenate((test_raa, test_pca1, test_pca2, test_pca3, test_pkx, test_pssm, test_hmm, test_onehot), axis=1)

    np.save('./multi-feature/' + ligand + '/bio_train.npy', train_bio)
    np.save('./multi-feature/' + ligand + '/bio_test.npy', test_bio)

if __name__ == '__main__':
    args = parse_args()
    if args.nucleic_acid not in ['DNA','RNA']:
        print('ERROR: nucleic_acid should be DNA or RNA')
        raise ValueError
    ligand = args.nucleic_acid
    train_dict = {'DNA': 'DNA-573_Train.txt',
                  'RNA': 'RNA-495_Train.txt'}
    test_dict = {'DNA': 'DNA-129_Test.txt',
                 'RNA': 'RNA-117_Test.txt'}
    Dataset_dir = './Datasets' + '/' + ligand
    train_dir = './Datasets' + '/' + ligand + '/{}'.format(train_dict[ligand])
    test_dir ='./Datasets' + '/' + ligand + '/{}'.format(test_dict[ligand])
    save_dir = './multi-feature/'+ligand

    print('read fasta file')
    train_list, test_list, train_seq, test_seq, train_label, test_label= read_fasta(train_dir, test_dir)

    print('generate pssm and hmm')
    generate_PSSM_HMM(train_list, test_list, ligand)

    print('gneerate one-hot')
    generate_one_hot(train_seq, test_seq)

    print('generate bio feature')
    generate_bio(ligand)

    print('generate label')
    generate_label(train_label, test_label, save_dir)

    print('generate dynamic_embdedding')
    train_dyna = get_dynamic_embdedding(train_seq)
    test_dyna = get_dynamic_embdedding(test_seq)
    np.save(save_dir + '/' + 'dyna_train.npy', np.array(train_dyna))
    np.save(save_dir + '/' + 'dyna_test.npy', np.array(test_dyna))

    print('generate generalization_embedding')
    train_vec = get_generalization_embedding(train_seq)
    test_vec = get_generalization_embedding(test_seq)
    np.save(save_dir + '/' + 'gene_train.npy', train_vec)
    np.save(save_dir + '/' + 'gene_test.npy', test_vec)
