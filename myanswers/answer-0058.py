import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def calcular_auc_logistico(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X, y)

    prob_pred = modelo.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, prob_pred)

    return float(auc)
