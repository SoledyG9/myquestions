import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def aplicar_pca(df, n_componentes):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca.explained_variance_ratio_
