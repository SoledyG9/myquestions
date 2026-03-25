import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

def generar_caso_de_uso_entrenar_modelo_estacional():
    # Generar fechas aleatorias en un rango
    n_samples = 30
    fechas = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Crear DataFrame con ruido y una variable inútil
    df = pd.DataFrame({
        'fecha': fechas,
        'variable_ruido': np.random.rand(n_samples), # Esta debería tender a cero en Lasso
        'promocion': np.random.choice([0, 1], n_samples)
    })
    
    # El target depende del día de la semana (estacionalidad)
    target = 'ventas'
    # Ventas más altas los fines de semana (5=Sábado, 6=Domingo)
    df[target] = df['fecha'].dt.dayofweek.map(lambda x: 100 if x >= 5 else 20) + np.random.normal(0, 5, n_samples)
    
    # --- Lógica interna para calcular el output esperado ---
    df_proc = df.copy()
    df_proc['fecha'] = pd.to_datetime(df_proc['fecha'])
    day = df_proc['fecha'].dt.dayofweek
    df_proc['dia_sin'] = np.sin(2 * np.pi * day / 7)
    df_proc['dia_cos'] = np.cos(2 * np.pi * day / 7)
    
    X = df_proc.drop(columns=['fecha', target])
    y = df_proc[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Lasso(alpha=0.1)
    model.fit(X_scaled, y)
    
    # Contar coeficientes eliminados por Lasso (cercanos a 0)
    coef_ceros = np.sum(np.isclose(model.coef_, 0, atol=1e-5))
    mae = mean_absolute_error(y, model.predict(X_scaled))
    
    input_dict = {
        "df": df,
        "target_col": target,
        "fecha_col": "fecha"
    }
    
    output = (model, int(coef_ceros), float(mae))
    
    return input_dict, output
