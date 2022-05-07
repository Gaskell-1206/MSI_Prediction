# Run aggregation by machine learning based methods
# Reference: Cao R, Yang F, Ma SC, Liu L, Zhao Y, Li Y, Wu DH, Wang T, Lu WJ, Cai WJ, Zhu HB, Guo XJ, Lu YW, Kuang JJ, Huan WJ, Tang WM, Huang K, Huang J, Yao J, Dong ZY. 
#            Development and interpretation of a pathomics-based model for the prediction of microsatellite instability in Colorectal Cancer. Theranostics 2020; 10(24):11080-11091. 
#            doi:10.7150/thno.49864. Available from http://www.thno.org/v10p11080.htm
#            The source codes of the referenced paper available at https://github.com/yfzon/EPLA.
# This code was modified by Ruoxun Zi for our work.

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
from xgboost.sklearn import XGBClassifier
import joblib
import pandas as pd
import csv
from numpy import save
import joblib
import pickle as pkl
from sklearn.naive_bayes import MultinomialNB
import sklearn.feature_extraction.text as ft

# PALHI: patch likelihood histogram
def genPatientIdxDict(patient_ID):
    ''' 
        generate the prediction list according to slide/patient name
    '''
    patient_idx_dict = {}
    unique_patient, unique_patient_idx = np.unique(patient_ID, return_index=True)
    for p in unique_patient:
        patient_idx_dict[p] = np.where(patient_ID == p)[0]

    return patient_idx_dict, unique_patient_idx


def loadLikelihood_test(llh_file):
    '''
        read the likelihood list according to tile name
        llh_file: likelihood file
    '''
    llh_tbl = pd.read_csv(llh_file, header=0, index_col=None)
  
    test_llh_tbl = llh_tbl.sort_values(by=['slides'])

    logging.info('We have {:} patients'.format(len(np.unique(test_llh_tbl['slides']))))

    te_data = {'patient_ID':test_llh_tbl['slides'].values,
               'patch_name':test_llh_tbl['tiles'].values,
               'likelihood':test_llh_tbl['probability'].values,
               'true_label':test_llh_tbl['target'].values}
    return te_data


def genLikelihoodHist(likelihood, patient_ID, num_bin, norm_hist = False):
    '''
        likelihood: (num_patch, )
        patient_ID: (num_patch, )
        num_bin: euqal size [0, 1]
        norm_hist: whether to normalize each hist
    return
        patient_hist: (num_unique_patient, num_bin)
        unique_patient_idx: (num_unique_patient, )
    '''
    bins = [-float('Inf')]
    bins.extend([i/num_bin for i in range(1, num_bin)])
    bins.append(float('Inf'))
    
    patient_idx_dict, unique_patient_idx = genPatientIdxDict(patient_ID)
    patient_hist = np.zeros((len(unique_patient_idx), num_bin))
    for i in range(len(unique_patient_idx)):
        idx = patient_idx_dict[patient_ID[unique_patient_idx[i]]]
        patient_hist[i,:] = np.histogram(likelihood[idx], bins = bins)[0]
        if norm_hist:
            patient_hist[i,:] = patient_hist[i,:] / np.sum(patient_hist[i,:])
    return patient_hist, unique_patient_idx


def genWsiDf_test(te_data, te_unique_patient_idx, te_pred_prob, te_pred_label):
    ''' columns: Sample.ID,  Patch.Num, WSI.Score, WSI.pred
    '''
    wsi_score = np.array(te_pred_prob)
    wsi_pred = np.array(te_pred_label)
    sample_ID = np.array(te_data['patient_ID'][te_unique_patient_idx])
    true_label = np.array(te_data['true_label'][te_unique_patient_idx])
    te_patch_num = np.zeros(len(te_unique_patient_idx), dtype=int)

    for i in range(len(te_unique_patient_idx)):
        idx = te_data['patient_ID'] == te_data['patient_ID'][te_unique_patient_idx[i]]
        te_patch_num[i] = np.sum(idx)

        
    patch_num = np.array(te_patch_num)

    wsi_pred_df = pd.DataFrame({'Sample.ID':sample_ID, 'Patch.Num':patch_num,
                                'WSI.Score':wsi_score, 'WSI.pred':wsi_pred, 'TrueLabel':true_label})

    return wsi_pred_df


def PALHI_inference(te_data, clf, num_bin=200, norm_hist=False):
    ''' PAtch Likelihood HIstogram pipeline
        tr_data, te_data: dict with 'patient_ID', 'MSI_label', 'MSI_score', 'patch_name', 'likelihood'
        cls_model:
        num_bin:
        norm_hist:
    '''
    te_patient_hist, te_unique_patient_idx = genLikelihoodHist(te_data['likelihood'], te_data['patient_ID'],
                                                               num_bin, norm_hist)

    te_pred_label = clf.predict(te_patient_hist)
    te_pred_prob = clf.predict_proba(te_patient_hist)[:,1]

    gc.collect()
    
    wsi_pred_df =  genWsiDf_test(te_data, te_unique_patient_idx, te_pred_prob, te_pred_label)

    return wsi_pred_df


# BoW: bag of words
def genBoW(data, precision):
    ''' data is a dict with 'patient_ID', 'patch_name', 'likelihood'
        precision: precision of BoW
    '''
    corpus_list = []
    sample_id_list = []
    patch_no_list = []
    
    patient_idx_dict, unique_patient_idx = genPatientIdxDict(data['patient_ID'])
    
    for i in range(len(unique_patient_idx)):
        pid = data['patient_ID'][unique_patient_idx[i]]
        llh = data['likelihood'][patient_idx_dict[pid]]
        llh = llh.tolist()
        words = ' '.join(["{0:.{1}f}".format(x, precision) for x in llh])
        wsi_patches = len(patient_idx_dict[pid])
        corpus_list.append(words)
        sample_id_list.append(pid)
        patch_no_list.append(wsi_patches)

    return unique_patient_idx, corpus_list, sample_id_list, patch_no_list


def BOW(te_data, cv, tf, precision, model):
    # generate corpus
    te_unique_patient_idx, corpus_te, sample_id_te, patch_no_te = genBoW(te_data, precision)
    
    test_tfmat = cv.transform(corpus_te)
    test_x = tf.transform(test_tfmat)

    logging.info("Results on testing set")

    te_pred_label = model.predict(test_x)
    te_pred_prob = model.predict_proba(test_x)[:, 1]

    wsi_pred_df =  genWsiDf_test(te_data, te_unique_patient_idx, te_pred_prob, te_pred_label)

    return wsi_pred_df

# Visulaization
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

if __name__ == "__main__":
    # main
    filename = 'prediction'
    form = filename + '.csv'
    te_data = loadLikelihood_test(form)
    # PALHI
    clf = XGBClassifier()
    clf = joblib.load(os.path.join('palhi.model'))
    wsi_pred_df_palhi = PALHI_inference(te_data, clf)
    # BoW
    model = joblib.load(os.path.join('bow.model'))
    cv = ft.CountVectorizer(decode_error="replace", vocabulary=pkl.load(open('bow_feature.pkl', "rb")))
    tf = pkl.load(open('bow_tfidftransformer.pkl', "rb"))
    wsi_pred_df_bow = BOW(te_data, cv, tf, 3, model)
    # ensemble
    weights = [0.5,0.5]
    youden_criterion = 0.5 #could be custom
    probability = weights[0]*wsi_pred_df_palhi['WSI.Score']+weights[1]*wsi_pred_df_bow['WSI.Score']
    prediction = probability.apply(lambda x: 1 if x >= youden_criterion else 0)
    modelPredDFs = pd.DataFrame({'slides':wsi_pred_df_palhi['Sample.ID'], 'target':wsi_pred_df_palhi['TrueLabel'], 'prediction':prediction, 'probability':probability})

    accuracy(modelPredDFs['target'],modelPredDFs['prediction'], filename+' ensemble')
    cm(modelPredDFs['target'],modelPredDFs['prediction'], filename+' ensemble')
    fpr, tpr, roc_auc = plot_roccurve(wsi_pred_df_palhi['TrueLabel'],wsi_pred_df_palhi['WSI.Score'], filename+' PALHI')

