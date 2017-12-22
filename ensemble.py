import pickle
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''
    
    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier = weak_classifier
        self.weak_classifiers = []
        for i in range(n_weakers_limit):
            self.weak_classifiers.append(weak_classifier(max_depth = 3))
        

    def is_good_enough(self):
        '''Optional'''
        pass


    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).
        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        print("newwwww")
        w = 1.0 / X.shape[0] * np.ones((X.shape[0]))
        self.alpha_array = np.empty((self.n_weakers_limit))
        
        for i in range(self.n_weakers_limit):
            self.weak_classifiers[i].fit(X, y, sample_weight = w)
            y_predict = np.sign(self.weak_classifiers[i].predict(X))

            eps = 0
            for j in range(y_predict.shape[0]):
                if (y_predict[j] != y[j][0]):
                    eps += w[j]
            print(eps)
            if (eps > 0.5):
                self.n_weakers_limit = i
                break
                
            self.alpha_array[i] = math.log((1 - eps) / eps) / 2
            
            Z = 0
            exp_array = np.empty(w.shape)
            for j in range(w.shape[0]):
                e = math.exp(-1.0 * self.alpha_array[i] * y[j][0] * y_predict[j])
                exp_array[j] = e
                Z += w[j] * e
                
            for j in range(w.shape[0]):
                w[j] = w[j] * exp_array[j] / Z
        

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.
        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        predict_result = np.zeros((X.shape[0]))
        
        for i in range(self.n_weakers_limit):
            y_predict = np.sign(self.weak_classifiers[i].predict(X))
            predict_result += y_predict * self.alpha_array[i]
        
        return predict_result
    

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.
        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.
        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        predict_result = self.predict_scores(X)
       
        for i in range(X.shape[0]):
            if (predict_result[i] >= threshold):
                predict_result[i] = 1
            elif (predict_result[i] < threshold):
                predict_result[i] = -1
            
        return predict_result


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)