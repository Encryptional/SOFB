import math
import numpy as np
from ensemble_DLsequence_net import ensemble_DLsequence_net
from utils import evaluate, label_sum, label_one_hot, split_data, load_data_vec
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import (EarlyStopping, LearningRateScheduler)
import os
import random
import warnings
import argparse
random.seed(2022)
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--nucleic_acid", dest="nucleic_acid", help="nucleic_acid type DNA and RNA")
    parser.add_argument("--epochs", dest="epochs", help="the max epoch")
    parser.add_argument("--batchsize", dest='batchsize', default=1024, help='batchsize')
    parser.add_argument("--ensemble", dest="ensemble",default=4, help="The number of ensemble number.")
    return parser.parse_args()

def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 7.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate

def train(positive_list_gen, positive_list_bio, positive_list_dyna, sub_list_gen, sub_list_bio, sub_list_dyna, batchsize, epochs, ensemble, callbacks):
    for i in range(ensemble):
        print("**********************************" + 'emsemble_model:' + str((i)) + "*****************************************")
        train_con = np.array(np.concatenate((sub_list_gen[i], positive_list_gen), axis=0))
        train_bio_con = np.array(np.concatenate((sub_list_bio[i], positive_list_bio), axis=0))
        trian_dyna_con = np.array(np.concatenate((sub_list_dyna[i], positive_list_dyna), axis=0))
        label_con = np.concatenate((np.zeros(len(sub_list_gen[i]), dtype=int), np.ones(len(positive_list_gen), dtype=int)))
        label_now = [str(i) for i in label_con]
        train_label_now = np.array(label_one_hot(label_now))
        ##split
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022)
        for train_index, val_index in split.split(train_con, train_label_now):
            train_X_vec = train_con[train_index]
            train_X_bio = train_bio_con[train_index]
            train_X_dyna = trian_dyna_con[train_index]
            val_X_xl = train_con[val_index]
            val_X_bio = train_bio_con[val_index]
            val_X_dyna = trian_dyna_con[val_index]
            train_y = train_label_now[train_index]
            val_y = train_label_now[val_index]
        ##model
        model = ensemble_DLsequence_net()
        model.fit([train_X_vec, train_X_bio, train_X_dyna], y=train_y,
                  epochs=epochs,
                  batch_size=batchsize,
                  callbacks=callbacks,
                  verbose=1,
                  validation_data=([val_X_xl, val_X_bio,val_X_dyna], val_y),
                  shuffle=True)
        model.save('./save_model/' + '{}'.format(ligand) + '/'+str(i)+'.h5')

def test(ligand, ensemble):
    ##test
    test_gen = np.load('./multi-feature/' + '{}'.format(ligand) + '/gene_test.npy').reshape(-1,1,1024)
    test_bio_vec = np.load('./multi-feature/' + '{}'.format(ligand) + '/bio_test.npy').reshape(-1,1,75)
    test_dyna = np.load('./multi-feature/' + '{}'.format(ligand) + '/dyna_test.npy').reshape(-1,1,1024)
    test_label = np.array(label_one_hot(np.load('./multi-feature/' + '{}'.format(ligand) + '/test_label.npy')))

    predict_result = [[0,0]]*len(test_label)
    for i in range(ensemble):
        model = ensemble_DLsequence_net()
        model.load_weights('./save_model/' + '{}'.format(ligand) + '/'+str(i)+'.h5')
        pro = model.predict([test_gen, test_bio_vec, test_dyna])
        predict_result = label_sum(predict_result, pro)

    print("%s\t%s\t%s\t%s\t%s" % ('Rec', 'Pre', 'F1', 'MCC', 'AUROC'))
    evaluate(test_label, predict_result)

if __name__ == '__main__':
    args = parse_args()
    ligand = args.nucleic_acid
    batchsize = int(args.batchsize)
    epochs = int(args.epochs)
    ensemble = int(args.ensemble)

    ##load data_vec
    train_gen, train_bio_vec, train_dyna = load_data_vec(ligand)

    positive_list_gen, positive_list_bio, positive_list_dyna, sub_list_gen, sub_list_bio, sub_list_dyna = split_data(train_gen, train_bio_vec, train_dyna, ensemble, ligand)

    callbacks = [EarlyStopping(monitor='val_loss', patience=6),LearningRateScheduler(step_decay)]

    # train(positive_list_gen, positive_list_bio, positive_list_dyna, sub_list_gen, sub_list_bio, sub_list_dyna, batchsize, epochs, ensemble, callbacks)

    test(ligand, ensemble)