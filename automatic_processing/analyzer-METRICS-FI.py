# -*-coding:Utf-8 -*

# Copyright: Marielle MALFANTE - GIPSA-Lab -
# Univ. Grenoble Alpes, CNRS, Grenoble INP, GIPSA-lab, 38000 Grenoble, France
# (04/2018)
#
# marielle.malfante@gipsa-lab.fr (@gmail.com)
#
# This software is a computer program whose purpose is to automatically
# processing time series (automatic classification, detection). The architecture
# is based on machine learning tools.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL

import json
from os.path import isfile, isdir
import datetime

import matplotlib.pyplot as plt

from features import FeatureVector
import pickle
import numpy as np
from DataReadingFunctions import requestObservation
from sklearn import preprocessing
from tools import butter_bandpass_filter
from featuresFunctions import energy, energy_u
from math import sqrt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from tools import print_cm
import time
from tools import extract_features
from copy import deepcopy
from sklearn.inspection import permutation_importance
import pandas as pd


class Analyzer:
    """ Object containing the needed tools to analyze a Dataset.
    It contains the features scaler, the model, and the labels encoder,
    and can be used to train a model from supervised data.
    - scaler: None if learn has not been called, the learnt scaler otherwise
    - model: None if learn has not been called, the learnt model otherwise
    - labelEncoder: None if learn has not been called, the label encoder (label proper translation
    to int) otherwise
    - pathToCatalogue: path to the labeling catalogue. /!\ Catalogue should have a definite format.
    Check out README for more information on the catalogue shape.
    - catalogue: loaded catalogue of labels
    - _verbatim: how chatty do you want your Analyze to be?
    """

    def __init__(self, config, verbatim=0):
        """
        Initialization method
        """
        self.scaler = None
        self.model = deepcopy(config.learning['algo'])
        self.labelEncoder = None
        self.pathToCatalogue = config.general['project_root']+config.application['name'].upper()+'/'+config.learning['path_to_catalogue']
        self.catalogue = pickle.load(open(self.pathToCatalogue,'rb'))
        self._verbatim = verbatim
        if self._verbatim>0:
            print('\n\n *** ANALYZER ***')
        return

    def __repr__(self):
        """
        Representation method (transform the object to str for display)
        """
        s = 'Analyzer object with model and scaler being: '+str(self.model)+' and ' +str(self.scaler)
        s += '\nCatalogue is at %s'%self.pathToCatalogue
        return s

    def learn(self, config, verbatim=None, forModelSelection=False, sss=None, model=None, featuresIndexes=None, returnData=True):
        """
        Method to train the analyzer.
        Labeled data are read from the catalogue, the data are preprocessed,
        features are extracted and scaled, and model is trained (with the
        standard labels).
        All the arguments with default values are for a "non classic" use of
        the analyzer object (model selection for example)
        Return None, but can return the data and labels if specified in returnData.
        """
        if verbatim is None:
            verbatim=self._verbatim

        # Get or define useful stuff
        features = FeatureVector(config, verbatim=verbatim)
        nData = len(self.catalogue.index)
        if returnData:
            allData = np.zeros((nData,), dtype=object)
        allLabels = np.zeros((nData,), dtype=object)
        allFeatures = np.zeros((nData, features.n_domains*features.n_features), dtype=float)

        # Read all labeled signatures (labels+data) from the catalogue, and extract features
        tStart = time.time()
        for i in range(len(self.catalogue.index)):
            if self._verbatim > 2:
                print('Data index: ', i)
            secondFloat = self.catalogue.iloc[i]['second']
            tStartSignature = datetime.datetime(int(self.catalogue.iloc[i]['year']),     \
                                                int(self.catalogue.iloc[i]['month']),    \
                                                int(self.catalogue.iloc[i]['day']),      \
                                                int(self.catalogue.iloc[i]['hour']),     \
                                                int(self.catalogue.iloc[i]['minute']),   \
                                                int(secondFloat), \
                                                int((secondFloat-int(secondFloat))*1000000)) #microseconds
            duration = self.catalogue.iloc[i]['length']
            path = self.catalogue.iloc[i]['path']
            (fs, signature) = requestObservation(config, tStartSignature, duration, path, verbatim=0)

            # If problem
            if len(signature) < 40:
                if verbatim > 2:
                    print('Data is not considered', tStartSignature)
                allFeatures[i] = None
                allLabels[i] = None
                continue

#            if returnData:
#                allData[i] = signature

            # Get label and check that it is single label (multi label not supported yet)
            lab = self.catalogue.iloc[i]['class']
            if type(lab) is list:
                print('Multi label not implemented for learning yet')
                return None
            allLabels[i] = lab

            # Filtering if needed
            f_min = self.catalogue.iloc[i]['f0']
            f_max = self.catalogue.iloc[i]['f1']
            if f_min and f_max:
                butter_order = config.analysis['butter_order']
                signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)

            # Preprocessing & features extraction
            allFeatures[i] = extract_features(config, signature.reshape(1, -1), features, fs)

        tEnd = time.time()
        if verbatim>0:
            print('Training data have been read and features have been extracted ', np.shape(allFeatures))
            print('Computation time: {} seconds'.format(tEnd-tStart))


        # Compress labels and features in case of None values (if reading is empty for example)
        i = np.where(allLabels != np.array(None))[0]
        allFeatures = allFeatures[i]
        allLabels = allLabels[i]
    #    if returnData:
    #        allData = allData[i]

        # Transform labels
        self.labelEncoder = preprocessing.LabelEncoder().fit(allLabels)
        allLabelsStd = self.labelEncoder.transform(allLabels)
        if verbatim>0:
            print('Model will be trained on %d classes'%len(self.labelEncoder.classes_), np.unique(allLabelsStd), self.labelEncoder.classes_)

        # Scale features and store scaler
        self.scaler = preprocessing.StandardScaler().fit(allFeatures)
        allFeatures = self.scaler.transform(allFeatures)
        if verbatim>0:
            print('Features have been scaled')

        # Get model from learning configuration file and learn
        self.model = deepcopy(config.learning['algo'])

        if forModelSelection:
            if model is None:
                pass
            else:
                self.model = model

        # Added by KBM (11/10/21)
        # Split into training and test sets
        # Beginning
        print('\nEvents per class:')
        print(self.catalogue['class'].value_counts())

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        for train_idx, test_idx in sss.split(allFeatures, allLabelsStd):
            # Save features and labels of each set as arrays
            # x ---> features
            # y ---> labels
            x_train, x_test = allFeatures[train_idx], allFeatures[test_idx]
            y_train, y_test = allLabelsStd[train_idx], allLabelsStd[test_idx]

        print('\nTraining on {} events. \nTesting on {} events.'.format(x_train.shape[0], x_test.shape[0]))
        # PLUS ALL CHANGES IN THE FOLLOWING TRAINING
        # Ending

        tStartLearning = time.time()
        print('\n TRAINING:')

        if featuresIndexes is None:
            # self.model = self.model.fit(allFeatures, allLabelsStd)
            self.model = self.model.fit(x_train, y_train)
        else:
            # self.model = self.model.fit(allFeatures[:, featuresIndexes], allLabelsStd)
            self.model = self.model.fit(allFeatures[:, train_idx], allLabelsStd[train_idx])
        tEndLearning = time.time()

        # Added by KBM
        # Beginning
        # Commenting Model evaluation
        #  Model Evaluation (a) with score, (b) with X-validation
        if verbatim>0:
            # NB: When model is trained (and evaluated by X-validation or score),
            # threshold is NOT used. Threshold is only used when the 'unknown'
            # class can occur (and this is obviously not the case with supervised
            # training)
            print('Model has been trained: ', self.model)
            print('Computation time (training): {} seconds'.format(tEndLearning-tStartLearning))

            if featuresIndexes is None:
                # allPredictions = self.model.predict(allFeatures)
                allPredictions = self.model.predict(x_train)
            else:
                # allPredictions = self.model.predict(allFeatures[:, featuresIndexes])
                allPredictions = self.model.predict(allFeatures[:, train_idx])

            # (a) Score evaluation
            # print('Model score is: ', accuracy_score(allLabelsStd, allPredictions))
            print('Model score is: ', accuracy_score(y_train, allPredictions))
            lab = list(range(len(self.labelEncoder.classes_)))  # 'unknown' class not needed.
            # CM = confusion_matrix(allLabelsStd,allPredictions,labels=lab)
            CM = confusion_matrix(y_train, allPredictions, labels=lab)
            print('and associated confusion matrix is:')
            print_cm(CM, list(self.labelEncoder.classes_), hide_zeroes=True, max_str_label_size=2, float_display=False)

            # (b) X-validation
            print('\nCross validation')
            sss2 = config.learning['cv']
            print(sss2)
            CM=list()
            acc=list()

            # Added by KBM
            # SCORES FOR X-VAL
            # Beginning
            EX_pre, EX_rec, EX_f1, EX_sup = [], [], [], []
            LP_pre, LP_rec, LP_f1, LP_sup = [], [], [], []
            RE_pre, RE_rec, RE_f1, RE_sup = [], [], [], []
            TR_pre, TR_rec, TR_f1, TR_sup = [], [], [], []
            VT_pre, VT_rec, VT_f1, VT_sup = [], [], [], []
            NO_pre, NO_rec, NO_f1, NO_sup = [], [], [], []
            # Ending

            model_Xval = deepcopy(self.model)
            # for (i, (train_index, test_index)) in enumerate(sss.split(allFeatures, allLabelsStd)):
            for (i, (train_index, test_index)) in enumerate(sss2.split(x_train, y_train)):
                # predictionsStd = model_Xval.fit(allFeatures[train_index], allLabelsStd[train_index]).predict(allFeatures[test_index])
                predictionsStd = model_Xval.fit(x_train[train_index], y_train[train_index]).predict(x_train[test_index])
                predictions = self.labelEncoder.inverse_transform(predictionsStd)
                true_lab_test_Xval = self.labelEncoder.inverse_transform(y_train[test_index])
                # CM.append(confusion_matrix(allLabels[test_index],predictions, labels=self.labelEncoder.classes_))
                CM.append(confusion_matrix(true_lab_test_Xval, predictions, labels=self.labelEncoder.classes_))
                # acc.append(accuracy_score(allLabels[test_index],predictions))
                acc.append(accuracy_score(true_lab_test_Xval, predictions))
                # Added by KBM
                # SCORES FOR X-VAL
                # Beginning
                class_report = classification_report(true_lab_test_Xval, predictions,
                                                     target_names=self.labelEncoder.classes_, output_dict=True)
                # Scores for explosions
                EX_pre.append(class_report['Explosion']['precision'])
                EX_rec.append(class_report['Explosion']['recall'])
                EX_f1.append(class_report['Explosion']['f1-score'])
                EX_sup.append(class_report['Explosion']['support'])
                # Scores for LPs
                LP_pre.append(class_report['LP']['precision'])
                LP_rec.append(class_report['LP']['recall'])
                LP_f1.append(class_report['LP']['f1-score'])
                LP_sup.append(class_report['LP']['support'])
                # Scores for Regionals
                RE_pre.append(class_report['Regional']['precision'])
                RE_rec.append(class_report['Regional']['recall'])
                RE_f1.append(class_report['Regional']['f1-score'])
                RE_sup.append(class_report['Regional']['support'])
                # Scores for tremors
                TR_pre.append(class_report['Tremor']['precision'])
                TR_rec.append(class_report['Tremor']['recall'])
                TR_f1.append(class_report['Tremor']['f1-score'])
                TR_sup.append(class_report['Tremor']['support'])
                # Scores for VTs
                VT_pre.append(class_report['VT']['precision'])
                VT_rec.append(class_report['VT']['recall'])
                VT_f1.append(class_report['VT']['f1-score'])
                VT_sup.append(class_report['VT']['support'])
                # Scores for noise
                NO_pre.append(class_report['noise']['precision'])
                NO_rec.append(class_report['noise']['recall'])
                NO_f1.append(class_report['noise']['f1-score'])
                NO_sup.append(class_report['noise']['support'])
                #Ending
            print('Cross-validation results: ', np.mean(acc)*100, ' +/- ', np.std(acc)*100, ' %')
            print_cm(np.mean(CM, axis=0),self.labelEncoder.classes_,hide_zeroes=True,max_str_label_size=2,float_display=False)
            # Added by KBM:
            # PRINT METRICS FOR EACH CLASS
            # Beginning
            print('\nMetrics for each class:')
            print('Class \tPrecision \tRecall \t\tF1-score \tSupport')
            print('Ex \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f}'.format(
                np.mean(EX_pre), np.std(EX_pre), np.mean(EX_rec), np.std(EX_rec),
                np.mean(EX_f1), np.std(EX_f1), np.mean(EX_sup), np.std(EX_sup)))
            print('LP \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f}'.format(
                np.mean(LP_pre), np.std(LP_pre), np.mean(LP_rec), np.std(LP_rec),
                np.mean(LP_f1), np.std(LP_f1), np.mean(LP_sup), np.std(LP_sup)))
            print('Re \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f}'.format(
                np.mean(RE_pre), np.std(RE_pre), np.mean(RE_rec), np.std(RE_rec),
                np.mean(RE_f1), np.std(RE_f1), np.mean(RE_sup), np.std(RE_sup)))
            print('Tr \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f}'.format(
                np.mean(TR_pre), np.std(TR_pre), np.mean(TR_rec), np.std(TR_rec),
                np.mean(TR_f1), np.std(TR_f1), np.mean(TR_sup), np.std(TR_sup)))
            print('VT \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f}'.format(
                np.mean(VT_pre), np.std(VT_pre), np.mean(VT_rec), np.std(VT_rec),
                np.mean(VT_f1), np.std(VT_f1), np.mean(VT_sup), np.std(VT_sup)))
            print('no \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f} \t{:.2f}+/-{:.2f}'.format(
                np.mean(NO_pre), np.std(NO_pre), np.mean(NO_rec), np.std(NO_rec),
                np.mean(NO_f1), np.std(NO_f1), np.mean(NO_sup), np.std(NO_sup)))
            # Ending
        # Ending

        # Added by KBM:
        # Beginning
        print('\nTESTING:')

        if featuresIndexes is None:
            test_predictions = self.model.predict(x_test)
        else:
            test_predictions = self.model.predict(allFeatures[:, test_idx])

        # Test evaluation
        print('Test score is: ', accuracy_score(y_test, test_predictions))
        lab = list(range(len(self.labelEncoder.classes_)))  # 'unknown' class not needed.
        CM_t = confusion_matrix(y_test, test_predictions, labels=lab)
        print('and associated confusion matrix is:')
        print_cm(CM_t, list(self.labelEncoder.classes_), hide_zeroes=True, max_str_label_size=2, float_display=False)

        print('\nMetrics for each class:')
        print(classification_report(y_test, test_predictions, target_names=self.labelEncoder.classes_))
        # Ending

        # Added by KBM (11/10/2021)
        # GET FEATURE IMPORTANCE
        # Beginning
        print("\nFEATURE IMPORTANCE")
        r = permutation_importance(self.model, x_test, y_test, n_repeats=30, random_state=2)
        for i in r.importances_mean.argsort()[::-1]:
            print(f"{features.featuresRef[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

        # Plot importances:
        sorted_idx = r.importances_mean.argsort()
        feature_importance = r.importances[sorted_idx]
        label_feature_importance = features.featuresRef[sorted_idx]
        # Data frame:
        df = pd.DataFrame(data=feature_importance)
        df.insert(loc=0, column='label', value=label_feature_importance)
        # Save dataframe as csv file
        file = 'feature_imp.csv'
        df.to_csv(file, index=False, header=False)
        # fig, ax = plt.subplots()
        # ax.boxplot(feature_importance.T, vert=False,
        #            labels=label_feature_importance)
        # ax.set_title("Permutation Importances (test set)")
        # fig.tight_layout()
        # fig.savefig('perm_importances.png', dpi=150)
        # Ending

        # Added by KBM
        # GRID SEARCH RESULTS:
        # Beginning
        # print("\nBest parameters set found on development set:")
        # print()
        # print(self.model.best_params_)
        # print()
        # print("Grid scores on training set:")
        # print()
        # means = self.model.cv_results_['mean_test_score']
        # stds = self.model.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, self.model.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # Ending

        if returnData:
            return allFeatures
    #        return allData, allLabels, allFeatures
        else:
            return None

    def save(self, config):
        """
        Method used to save the object for later use (depending on the
        application, training can take a while and you might want to save the analyzer)
        """
        path = config.general['project_root'] + config.application['name'].upper() + '/res/' + config.configuration_number + '/' + config.general['path_to_res']
        savingPath = path+'analyzer'
        pickle.dump(self.__dict__, open(savingPath, 'wb'), 2)
        if self._verbatim > 0:
            print('Analyzer has been saved at: ', savingPath)
        return

    def load(self, config):
        """
        Method used to load the object.
        """
        verbatim = self._verbatim
        path = config.general['project_root'] + config.application['name'].upper() + '/res/' + config.configuration_number + '/' + config.general['path_to_res']
        savingPath = path+'analyzer'
        tmp_dict = pickle.load(open(savingPath,'rb'))
        self.__dict__.update(tmp_dict)
        self._verbatim = verbatim
        if self._verbatim > 0:
            print('Analyzer has been loaded from: ', savingPath)
        return
