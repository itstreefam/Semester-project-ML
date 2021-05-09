import logging
import traceback
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

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

        dataset['heart_disease'] = dataset['heart_disease'].replace(1, 0)
        dataset['heart_disease'] = dataset['heart_disease'].replace(2, 1)

        X = dataset.drop(columns=['heart_disease'])
        y = dataset['heart_disease'].values.reshape(X.shape[0], 1)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

        svm = SVM(max_iteration=100, kernel_type='linear', regularization=1.0, learning_rate=0.001, tol=1e-5)
        svm.train(Xtrain, ytrain)

        test_pred = svm.predict(Xtest)
        i = round(0.25 * len(test_pred))
        test_pred = np.array(test_pred[:i])
        print("Test accuracy is {}".format(accuracy_score(ytest.reshape(-1, ), test_pred)))

    except Exception as e:
        logging.error(traceback.format_exc())