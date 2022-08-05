#!/usr/bin/env python
# coding: utf-8

# In[33]:


import argparse
import os
import sys
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV


try:
    from sklearn.externals import joblib
except:
    import joblib


def run(arguments):
    test_file = None
    train_file = None
    validation_file = None
    joblib_file = "LR_model.pkl"


    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-e', '--test', help='Test attributes (to predict)')
    group1.add_argument('-n', '--train', help='Train data')
    parser.add_argument('-v', '--validation', help='Validation data')

    args = parser.parse_args(arguments)

    Train = False
    Test = False
    Validation = False

    if args.test != None:
        Test = True
            
    else:
        if args.train != None:
            print(f"Training data file: {args.train}")
            Train = True

        if args.validation != None:
            print(f"Validation data file: {args.validation}")
            Validation = True

    if Train and Validation:
        file_train = pd.read_csv(args.train,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
    # real review? 1=real review, 0=fake review
    # Category: Type of product
    # Product rating: Rating given by user
    # Review text: What reviewer wrote

        # Create TfIdf vector of review using 5000 words as features
        vectorizer = TfidfVectorizer(max_features=5000)
        # Transform text data to list of strings
        corpora = file_train['text_'].astype(str).values.tolist()
        # Obtain featurizer from data
        vectorizer.fit(corpora)
        # Create feature vector
        X = vectorizer.transform(corpora)

        print("Words used as features:")
        try:
            print(vectorizer.get_feature_names_out())
        except:
            print(vectorizer.get_feature_names())

        # Saves the words used in training
        with open('vectorizer.pk', 'wb') as fout:
            pickle.dump(vectorizer, fout)

        file_validation = pd.read_csv(args.validation,quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})

        best_accuracy = 0

        # TODO: The following code is performing regularization incorrectly.
        # Your goal is to fix the code.
        for C in [100,10,1,0.1,0.01,0.001]:
            lr = LogisticRegressionCV(penalty="l1", tol=0.001, C=C, fit_intercept=True, solver="saga", intercept_scaling=1, random_state=42)
            # You can safely ignore any "ConvergenceWarning" warnings
            lr.fit(X.toarray(), file_train['real review?'])
            # Get logistic regression predictions
            y_hat = lr.predict_proba(X.toarray())[:,1]

            y_pred = (y_hat > 0.5) + 0 # + 0 makes it an integer

            # Accuracy of predictions with the true labels and take the percentage
            # Because our dataset is balanced, measuring just the accuracy is OK
            accuracy = (y_pred == file_train['real review?']).sum() / file_train['real review?'].size
            print(f'Accuracy {accuracy}')
            print(f'Fraction of non-zero model parameters {np.sum(lr.coef_==0)+1}')
        
            if accuracy > best_accuracy:
                # Save logistic regression model
                joblib.dump(lr, joblib_file)
                best_accuracy = accuracy


    elif Test:
        # This part will be used to apply your model to the test data
        vectorizer = pickle.load(open('vectorizer.pk', 'rb'))
            
        # Read test file
        file_test = pd.read_csv(args.test,quotechar='"',usecols=[0,1,2,3,4],dtype={'ID':int,'real review?': int,'category': str, 'rating': int, 'text_': str})
        # Transform text into list of strigs
        corpora = file_test['text_'].astype(str).values.tolist()
        # Use the words obtained in training to encode in testing
        X = vectorizer.transform(corpora)

        # Load trained logistic regression model
        lr = joblib.load(joblib_file)

        # Competition evaluation is AUC... what is the correct output for AUC evaluation?
        y_hat = lr.predict_proba(X.toarray())[:,1]

        y_pred = (y_hat > 0.5)+0 # + 0 makes it an integer

        print(f"ID,real review?")
        for i,y in enumerate(y_pred):
            print(f"{i},{y}")


    else:
        print("Training requires both training and validation data files. Test just requires test attributes.")

        
if __name__ == "__main__":
    run(sys.argv[1:])
