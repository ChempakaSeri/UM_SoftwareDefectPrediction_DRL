#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:41:01 2020

@author: chempakaseri
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import joblib
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score

features = pd.read_csv("C:/Users/Ameer/Documents/sdp/data/platform_v1.csv")
x, y = features.iloc[:,:-1],features.iloc[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3, 
                                                    random_state=123)

##  ------------------ use DMatrix for xgboost ------------------------------
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

##  ----------------- use SVM file for xgboost ------------------------------
dump_svmlight_file(x_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(x_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

##  ------------------- set xgboost parameters ------------------------------
param = {
            'max_depth' : 3,                # the max depth of each tree
            'eta' : 0.3,                    # the training step of each iteration
            'silent' : 1,                   # logging mode = quiet
            'objective' : 'multi:softprob', # error evaluation for multiclass training
            'num_class' : 2                 # the number of classes that exist in the dataset
        }

num_round = 20 # iteration

##  ------------------------ numpy array ------------------------------------
#   training and testing - numpy metrics
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

#   extracting the most confident predictions
best_preds = np.asarray([np.argmax(line) for line in preds])


print ("Numpy array accuracy: ", accuracy_score(y_test, best_preds))
print ("Numpy array precision: ", precision_score(y_test, best_preds, average='weighted'))
print ("Numpy array recall: ", recall_score(y_test, best_preds, average='weighted'))
print ("Numpy array f-measure: ", f1_score(y_test, best_preds, average='weighted'))

##  ---------------- confusion matrix ----------------------------------------

#importing confusion matrix
y_pred = best_preds

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.4f}'.format(f1_score(y_test, y_pred, average='weighted')))


from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2']))


##  ------------------------ svm file ----------------------------------------

#   training and testing - svm file
#bst_svm = xgb.train(param, dtrain_svm), num_round
#preds = bst.predict(dtest_svm)

#   extracting most confident predictions
#best_preds_svm = [np.argmax(line) for line in preds]
#print ("SVM file precision:", precision_score(y_test, best_preds_svm, average='macro'))


##  ------------------------ dump the model ----------------------------------
#bst.dump_model('/Users/chempakaseri/Spyder/xgboost/muhaimin/Model/dump.raw.txt')
#bst_svm.dump_model('dump_svm.raw.txt')

##  -------------------- save the model for later ----------------------------
#joblib.dump(bst,open('/Users/chempakaseri/Spyder/xgboost/0611/bst_model.pkl','wb'))
#bst.save_model('/Users/chempakaseri/Spyder/xgboost/.pkl')
#joblib.dump(bst_svm,'bst_svm_model.pkl', compress=True)



