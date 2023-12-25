import math
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
import random

def load_data_vec(ligand):
    train_T5 = np.load('./multi-feature/' + '{}'.format(ligand) + '/ProtT5_train.npy').reshape(-1, 1, 1024)
    train_bio_vec = np.load('./multi-feature/' + '{}'.format(ligand) + '/bio_train.npy').reshape(-1, 1, 75)
    train_NABert = np.load('./multi-feature/' + '{}'.format(ligand) + '/NABert_train.npy').reshape(-1, 1, 1024)
    return train_gen, train_bio_vec, train_dyna

def label_one_hot(label_list):
    label = []
    for i in label_list:
        if i=='0':
            label.append([1,0])
        else:
            label.append([0,1])
    return label

def softmax(x, axis=1):
    # 计算每行的最大值
    row_max1 = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max1 = row_max1.reshape(-1, 1)
    x = x - row_max1

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def evaluate(y_true_prob,y_pred_prob):
  if y_pred_prob[0][1]+y_pred_prob[0][0] != 1.0:
    y_pred_prob = softmax(np.array(y_pred_prob))
  y_pred = np.argmax(y_pred_prob,axis=1)
  y_true = np.argmax(y_true_prob,axis=1)
  mcc = matthews_corrcoef(y_true,y_pred)
  pre = precision_score(y_true,y_pred)
  recall = recall_score(y_true,y_pred)
  f1 = f1_score(y_true,y_pred)
  auroc = roc_auc_score(y_true, y_pred_prob[:,1])
  print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (recall, pre, f1, mcc, auroc))
  return len(y_pred), recall, pre, f1, mcc, auroc

def label_sum(pre, now):
    c = []
    for i in range(len(now)):
        c.append(np.sum((pre[i], now[i]), axis=0))
    return c

def split_data(train_gen, train_bio_vec, train_dyna, emsemble_num, ligand):
    label = np.load('./multi-feature/' + '{}'.format(ligand) + '/train_label.npy')
    negative_list = []
    positive_list = []
    for i in range(len(label)):
        if label[i] == '0':
            negative_list.append(i)
        else:
            positive_list.append(i)
    random.shuffle(negative_list)
    split_num = emsemble_num
    sample_num = list(label).count('0') // split_num
    sub_list_gen = []
    sub_list_bio = []
    sub_list_dyna = []
    positive_list_gen = train_gen[positive_list]
    positive_list_bio = train_bio_vec[positive_list]
    positive_list_dyna = train_dyna[positive_list]
    for i in range(split_num):
        start = i * sample_num
        end = (i + 1) * sample_num
        if i == split_num - 1:
            end = len(negative_list) ##不平均
        sub_list_gen.append(train_gen[negative_list[start:end]])
        sub_list_bio.append(train_bio_vec[negative_list[start:end]])
        sub_list_dyna.append(train_dyna[negative_list[start:end]])
    return positive_list_gen, positive_list_bio,positive_list_dyna, sub_list_gen, sub_list_bio, sub_list_dyna
