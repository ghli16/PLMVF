import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

class MLModels(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rf_pipeline = Pipeline([('rf', RandomForestClassifier(random_state=42, n_jobs=-1))])
        self.svm_pipeline = Pipeline([('svm', SVC(probability=True, random_state=42))])
        self.xgb_pipeline = Pipeline([('xgb', XGBClassifier(random_state=42, n_jobs=-1))])
        self.mlp_pipeline = Pipeline([('mlp', MLPClassifier(random_state=42))])
    def fit(self, X, y):
        self.rf_pipeline.fit(X, y)
        self.svm_pipeline.fit(X, y)
        self.xgb_pipeline.fit(X, y)
        self.mlp_pipeline.fit(X, y)
        return self

    def transform(self, X):
        rf_probs = self.rf_pipeline.predict_proba(X)
        svm_probs = self.svm_pipeline.predict_proba(X)
        xgb_probs = self.xgb_pipeline.predict_proba(X)
        mlp_probs = self.mlp_pipeline.predict_proba(X)
        return np.hstack((rf_probs, svm_probs, xgb_probs, mlp_probs))