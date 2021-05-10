import logging
import traceback
import warnings

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(1234)

from svm import SVM

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    try:
        print('Statlog (Heart) Data Set')
        columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
                   'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
                   'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', "slope of the peak",
                   'num_of_major_vessels', 'thal', 'heart_disease']
        dataset = pd.read_csv("data/heart.dat", sep=' ', names=columns)
        dataset.dropna(axis="columns", how="any", inplace=True)

        dataset['heart_disease'] = dataset['heart_disease'].replace(1, -1)
        dataset['heart_disease'] = dataset['heart_disease'].replace(2, 1)

        X = dataset.drop(columns=['heart_disease'])
        y = dataset['heart_disease'].values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

        svm = SVM(max_iteration=100, kernel_type='linear', regularization=10, learning_rate=0.01, tol=1e-5)
        svm.train(Xtrain, ytrain)

        train_pred = svm.predict(Xtrain)
        test_pred = svm.predict(Xtest)

        svm.info()
        print("Train accuracy is {}".format(accuracy_score(ytrain, train_pred)))
        print("Test accuracy is {}".format(accuracy_score(ytest, test_pred)))

        print('Breast Cancer Wisconsin (Diagnostic) Data Set')
        dataset2 = pd.read_csv("data/breast_cancer_wisconsin_diagnostic_data.csv")
        dataset2.dropna(axis="columns", how="any", inplace=True)

        dataset2['diagnosis'] = dataset2['diagnosis'].replace('B', -1)
        dataset2['diagnosis'] = dataset2['diagnosis'].replace('M', 1)

        X2 = dataset2.drop(columns=['id', 'diagnosis'])
        y2 = dataset2['diagnosis'].values
        Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(X2, y2, test_size=0.2, random_state=0)

        svm2 = SVM(max_iteration=100, kernel_type='linear', regularization=5, learning_rate=0.01, tol=1e-5)
        svm2.train(Xtrain2, ytrain2)

        train_pred2 = svm2.predict(Xtrain2)
        test_pred2 = svm2.predict(Xtest2)

        svm2.info()
        print("Train accuracy is {}".format(accuracy_score(ytrain2, train_pred2)))
        print("Test accuracy is {}".format(accuracy_score(ytest2, test_pred2)))

    except Exception as e:
        logging.error(traceback.format_exc())