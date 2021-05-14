import logging
import traceback
import warnings
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
np.random.seed(1234)

from svm2 import SVM, MultiSVM

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

        svm = SVM(max_iteration=100, kernel_type='rbf', regularization=10, learning_rate=0.01, tol=0.001)

        start = time.time()
        svm.train(Xtrain, ytrain)
        end = time.time()

        train_pred = svm.predict(Xtrain)
        test_pred = svm.predict(Xtest)

        svm.info()
        print("Train time is {}".format(end - start))
        print("Train accuracy is {}".format(accuracy_score(ytrain, train_pred)))
        print("Test accuracy is {}".format(accuracy_score(ytest, test_pred)))

        # Sklearn library implementation
        clf = SVC(kernel='linear', C=10)
        start = time.time()
        clf.fit(Xtrain, ytrain)
        end = time.time()
        train_lib_pred = clf.predict(Xtrain)
        test_lib_pred = clf.predict(Xtest)
        print("Train time from sklearn is {}".format(end - start))
        print("Train accuracy from sklearn is {}".format(accuracy_score(ytrain, train_lib_pred)))
        print("Test accuracy from sklearn is {}".format(accuracy_score(ytest, test_lib_pred)))

        print('\n')
        print('Breast Cancer Wisconsin (Diagnostic) Data Set')
        dataset2 = pd.read_csv("data/breast_cancer_wisconsin_diagnostic_data.csv")
        dataset2.dropna(axis="columns", how="any", inplace=True)

        dataset2['diagnosis'] = dataset2['diagnosis'].replace('B', -1)
        dataset2['diagnosis'] = dataset2['diagnosis'].replace('M', 1)

        X2 = dataset2.drop(columns=['id', 'diagnosis'])
        y2 = dataset2['diagnosis'].values
        Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(X2, y2, test_size=0.2, random_state=0)

        svm2 = SVM(max_iteration=100, kernel_type='rbf', regularization=5, learning_rate=0.01, tol=1e-5)
        start = time.time()
        svm2.train(Xtrain2, ytrain2)
        end = time.time()

        train_pred2 = svm2.predict(Xtrain2)
        test_pred2 = svm2.predict(Xtest2)

        svm2.info()
        print("Train time is {}".format(end - start))
        print("Train accuracy is {}".format(accuracy_score(ytrain2, train_pred2)))
        print("Test accuracy is {}".format(accuracy_score(ytest2, test_pred2)))

        # Sklearn library implementation
        clf2 = SVC(kernel='linear', C=5)
        start = time.time()
        clf2.fit(Xtrain2, ytrain2)
        end = time.time()
        train_lib_pred2 = clf2.predict(Xtrain2)
        test_lib_pred2 = clf2.predict(Xtest2)
        print("Train time from sklearn is {}".format(end - start))
        print("Train accuracy from sklearn is {}".format(accuracy_score(ytrain2, train_lib_pred2)))
        print("Test accuracy from sklearn is {}".format(accuracy_score(ytest2, test_lib_pred2)))

        print('\n')
        print('Red Wine Quality Data Set')
        dataset3 = pd.read_csv("data/red_wine_quality.csv")
        dataset3.dropna(axis="columns", how="any", inplace=True)

        dataset3['quality'] = dataset3['quality'].replace([3, 4, 5], -1)
        dataset3['quality'] = dataset3['quality'].replace([6, 7, 8], 1)

        X3 = dataset3.drop(columns=['quality'])
        y3 = dataset3['quality'].values
        Xtrain3, Xtest3, ytrain3, ytest3 = train_test_split(X3, y3, test_size=0.2, random_state=21)

        svm3 = SVM(max_iteration=100, kernel_type='rbf', regularization=10, learning_rate=0.001, tol=1e-5)
        start = time.time()
        svm3.train(Xtrain3, ytrain3)
        end = time.time()

        train_pred3 = svm3.predict(Xtrain3)
        test_pred3 = svm3.predict(Xtest3)

        svm3.info()
        print("Train time is {}".format(end - start))
        print("Train accuracy is {}".format(accuracy_score(ytrain3, train_pred3)))
        print("Test accuracy is {}".format(accuracy_score(ytest3, test_pred3)))

        # Sklearn library implementation
        clf3 = SVC(kernel='rbf', C=10)
        start = time.time()
        clf3.fit(Xtrain3, ytrain3)
        end = time.time()
        train_lib_pred3 = clf3.predict(Xtrain3)
        test_lib_pred3 = clf3.predict(Xtest3)
        print("Train time from sklearn is {}".format(end - start))
        print("Train accuracy from sklearn is {}".format(accuracy_score(ytrain3, train_lib_pred3)))
        print("Test accuracy from sklearn is {}".format(accuracy_score(ytest3, test_lib_pred3)))

    except Exception as e:
        logging.error(traceback.format_exc())

