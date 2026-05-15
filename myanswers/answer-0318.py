import pandas as pd


def detectar_cambios_percentuales(df, grupo_col, tiempo_col, valor_col, umbral):
    df_resultado = df.copy()

    df_resultado = df_resultado.sort_values(
        [grupo_col, tiempo_col]
    ).reset_index(drop=True)

    df_resultado["cambio_pct"] = (
        df_resultado.groupby(grupo_col)[valor_col]
        .pct_change()
    )

    df_resultado = (
        df_resultado[df_resultado["cambio_pct"].notna()]
        .loc[lambda x: x["cambio_pct"].abs() > umbral]
        .reset_index(drop=True)
    )

    return df_resultado
