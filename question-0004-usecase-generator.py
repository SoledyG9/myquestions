import pandas as pd
import numpy as np

def generar_caso_de_uso_analizar_eficiencia_rutas():
    n = np.random.randint(5, 10)
    umbral = np.random.uniform(0.1, 0.2)
    df_in = pd.DataFrame({
        'ruta_id': [f'R_{i}' for i in range(n)],
        'distancia_km': np.random.uniform(50, 200, n),
        'litros_consumidos': np.random.uniform(5, 40, n)
    })
    # Lógica esperada
    df_out = df_in.copy()
    df_out['consumo_por_km'] = df_out['litros_consumidos'] / df_out['distancia_km']
    df_out = df_out[df_out['consumo_por_km'] > umbral].copy()
    if not df_out.empty:
        df_out['exceso'] = (df_out['consumo_por_km'] - umbral) * df_out['distancia_km']
