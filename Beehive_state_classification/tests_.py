import os
import pdb
import sys
import numpy as np
import pandas as pd
import glob
from utilsBeehiveState import report_SVM_beehiveState_results, SVM_Classification_BeehiveSTATE

sys.path.append('/Users/annayang/Documents/QueenBee/Bee_NotBee_classification')

from utils import load_audioFiles_saves_segments, write_Statelabels_from_samplesFolder, get_samples_id_perSet, get_features_from_samples, get_GT_labels_fromFiles, labels2binary, get_list_samples_names, split_samples_ramdom

"""
# CODE FOR OLD DATA
old_path_save_audio_labels='/Users/annayang/Documents/QueenBee/Dataset'+os.sep  # path where to save audio segments and labels files.
old_path_raw_state_labels = '/Users/annayang/Documents/QueenBee/Dataset/old_state_labels.csv'

print('beginning feature extraction...')
old_sample_ids_test, old_sample_ids_train, old_sample_ids_val = get_samples_id_perSet(old_path_save_audio_labels+'split_random_0.json')
old_X_train = get_features_from_samples(old_path_save_audio_labels, old_sample_ids_train, 'MFCCs20', 'NO', 0)
old_X_val = get_features_from_samples(old_path_save_audio_labels, old_sample_ids_val, 'MFCCs20', 'NO', 0)
old_X_test = get_features_from_samples(old_path_save_audio_labels, old_sample_ids_test, 'MFCCs20', 'NO', 0)

print('reshaping X arrays...')
nsamples, nx, ny = np.array(old_X_train).shape
old_X_train = np.array(np.array(old_X_train).reshape((nsamples,nx*ny)))

nsamples, nx, ny = np.array(old_X_val).shape
old_X_val = np.array(np.array(old_X_val).reshape((nsamples,nx*ny)))

nsamples, nx, ny = np.array(old_X_test).shape
old_X_test = np.array(np.array(old_X_test).reshape((nsamples,nx*ny)))

print('creating Y arrays...')
labels2read = 'state_labels'
labels_train = get_GT_labels_fromFiles(old_path_save_audio_labels, old_sample_ids_train, labels2read)
old_Y_train= labels2binary('missing queen', labels_train)

labels_val = get_GT_labels_fromFiles(old_path_save_audio_labels, old_sample_ids_val, labels2read)
old_Y_val= labels2binary('missing queen', labels_val)

labels_test = get_GT_labels_fromFiles(old_path_save_audio_labels, old_sample_ids_test, labels2read)
old_Y_test = labels2binary('missing queen', labels_test)

X_flat_train=np.concatenate((np.array(old_X_train) , np.array(old_X_val)))
Y_flat_train=np.concatenate((np.array(old_Y_train) , np.array(old_Y_val)))

"""
# CODE FOR NEW DATA
path_save_audio_labels='/Users/annayang/Documents/QueenBee/MyDataset'+os.sep  # path where to save audio segments and labels files.
path_raw_state_labels = '/Users/annayang/Documents/QueenBee/MyDataset/state_labels.csv'

print('beginning feature extraction...')
sample_ids_test, sample_ids_train, sample_ids_val = get_samples_id_perSet(path_save_audio_labels+'split_random_0.json')
X_train = get_features_from_samples(path_save_audio_labels, sample_ids_train, 'MFCCs20', 'NO', 0)
X_val = get_features_from_samples(path_save_audio_labels, sample_ids_val, 'MFCCs20', 'NO', 0)
X_test = get_features_from_samples(path_save_audio_labels, sample_ids_test, 'MFCCs20', 'NO', 0)

print('reshaping X arrays...')
nsamples, nx, ny = np.array(X_train).shape
X_train = np.array(np.array(X_train).reshape((nsamples,nx*ny)))

nsamples, nx, ny = np.array(X_val).shape
X_val = np.array(np.array(X_val).reshape((nsamples,nx*ny)))

nsamples, nx, ny = np.array(X_test).shape
X_test = np.array(np.array(X_test).reshape((nsamples,nx*ny)))

print('creating Y arrays...')
labels2read = 'state_labels'
labels_train = get_GT_labels_fromFiles(path_save_audio_labels, sample_ids_train, labels2read)
Y_train= labels2binary('missing queen', labels_train)

labels_val = get_GT_labels_fromFiles(path_save_audio_labels, sample_ids_val, labels2read)
Y_val= labels2binary('missing queen', labels_val)

labels_test = get_GT_labels_fromFiles(path_save_audio_labels, sample_ids_test, labels2read)
Y_test= labels2binary('missing queen', labels_test)

# X_flat_test=np.concatenate((np.array(X_train), np.array(X_val), np.array(X_test)))
# Y_flat_test=np.concatenate((np.array(Y_train), np.array(Y_val), np.array(Y_test)))



# audiofilenames_list = [os.path.basename(x) for x in glob.glob(path_audioFiles+'*.mp3')]
# audiofilenames_list.extend([os.path.basename(x) for x in glob.glob(path_audioFiles+'*.wav')])


# path_samplesFolder = path_save_audio_labels
# path_save = path_samplesFolder 
# reads names of samples (.wav files) and creates corresponding states label file.
# raw_state_labels = pd.read_csv(path_raw_state_labels)
# write_Statelabels_from_samplesFolder(path_save, path_samplesFolder, raw_state_labels)




#print(len(X_flat_train))
# print(len(X_flat_test))


#print(len(Y_flat_train))
# print(len(Y_flat_test))


clf, Test_GroundT, Train_GroundT, Test_Preds, Train_Preds, Test_Preds_Proba, Train_Preds_Proba = SVM_Classification_BeehiveSTATE(X_train, Y_train, X_test, Y_test)
summary_filename = 'summary'
path_results = '/Users/annayang/Documents/QueenBee/results/'

# report_SVM_beehiveState_results(summary_filename, path_results, clf, Test_GroundT, Train_GroundT, Test_Preds, Train_Preds, Test_Preds_Proba, Train_Preds_Proba, 'pretrained', audiofilenames_list,  60)

# pdb.set_trace()

