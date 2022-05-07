# Run aggregation by majority vote
# Author: Ruoxun Zi

# if run at google colab
# from google.colab import drive
# drive.mount('/content/drive')

import os
import gc
import time
import logging
import numpy as np
import pandas as pd
import argparse
import pandas as pd
import csv
from numpy import save

def loadResults_test1(llh_file):
    '''
        read the prediction list according to tile name
    '''
    llh_tbl = pd.read_csv(llh_file, header=0, index_col=None)
  
    test_llh_tbl = llh_tbl.sort_values(by=['slides'])

    logging.info('We have {:} patients'.format(len(np.unique(test_llh_tbl['slides']))))

    te_data = {'patient_ID':test_llh_tbl['slides'].values,
               'patch_name':test_llh_tbl['tiles'].values,
               'likelihood':test_llh_tbl['probability'].values,
               'prediction':test_llh_tbl['prediction'].values,
               'true_label':test_llh_tbl['target'].values}
    return te_data


def genPatientIdxDict1(df):
    ''' 
        generate the prediction list according to slide/patient name
    '''
    patient_idx_dict = {}
    likelihood_list = {}
    prediction_list = {}
    unique_patient, unique_patient_idx = np.unique(df['patient_ID'], return_index=True)
    patient_num = np.zeros((len(unique_patient_idx),))
    patient_avg = np.zeros((len(unique_patient_idx),))
    patient_max = np.zeros((len(unique_patient_idx),))
    true_label = np.zeros((len(unique_patient_idx),))
    pred_label = np.zeros((len(unique_patient_idx),))
    prob = np.zeros((len(unique_patient_idx),))
    for p in range(len(unique_patient)):
        patient_idx_dict[unique_patient[p]] = np.where(df['patient_ID'] == unique_patient[p])[0]
        likelihood_list[unique_patient[p]] = df['likelihood'][patient_idx_dict[unique_patient[p]]]
        prediction_list[unique_patient[p]] = df['prediction'][patient_idx_dict[unique_patient[p]]]
        patient_num[p] = len(patient_idx_dict[unique_patient[p]])
        patient_avg[p] = np.median(prediction_list[unique_patient[p]])
        patient_max[p] = np.max(prediction_list[unique_patient[p]])
        true_label[p] = df['true_label'][patient_idx_dict[unique_patient[p]][0]]
        prob[p] = np.median(likelihood_list[unique_patient[p]])
        if patient_avg[p] > 0.5:
          pred_label[p] = 1

    return likelihood_list, patient_num, patient_avg, patient_max, true_label, pred_label, prob


import itertools    
from sklearn.metrics import  confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def cm(y_true, y_pred, name):
    ''' 
        calculate and draw confusion matrix
    '''
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    class_names = ['MSS','MSI']
    plt.imshow(cnf_matrix, interpolation='nearest', cmap = plt.cm.Blues)
    plt.colorbar()
    plt.title(name, fontsize = 16)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
      plt.text(j, i, format(cnf_matrix[i, j], fmt),
               horizontalalignment="center",
               color="white" if cnf_matrix[i, j] > thresh else "black", fontsize = 16)
      plt.tight_layout()
      plt.ylabel('True label', fontsize = 16)
      plt.xlabel('Predicted label', fontsize = 16)

def plot_roccurve(y_true, y_pred, name):
    ''' 
        calculate and plot ROC curve
    '''
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = {0:0.2f})".format(roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize = 16)
    plt.show()

    return fpr, tpr, roc_auc

def accuracy(y_true, y_pred, name):
    ''' 
        calculate and print accuracy
    '''
    print('Accuracy', (name), ': ',len(np.where(y_true==y_pred)[0])/len(y_true))

# main
if __name__ == "__main__":
    filename = 'prediction'
    form = filename + '.csv'
    df = loadResults_test1(form)
    likelihood_list, patient_num, patient_avg, patient_max, true_label, pred_label, prob = genPatientIdxDict1(df)

    accuracy(true_label, pred_label, filename)

    cm(true_label, pred_label, filename)

    fpr, tpr, roc_auc = plot_roccurve(true_label, prob, filename)

