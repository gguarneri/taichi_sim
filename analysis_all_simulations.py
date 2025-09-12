import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

arquivos_txt_path = os.path.join(".", "ensaios", "ponto", "results", "result_*__desc.txt")
arquivos_txt = glob.glob(arquivos_txt_path)


def extrair_dados(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as f:
        texto = f.read()

    simulador = re.search(r"Simulador:\s*(.+)", texto)
    tempo_medio = re.search(r"Tempo medio total \(inclui transferencia de dados\):\s*([\d\.]+)s", texto)
    tempo_std = re.search(r"Tempo medio total \(inclui transferencia de dados\):.*?Desvio padrao:\s*([\d\.eE\-]+)", texto, re.DOTALL)

    mse_campo = re.search(r"MSE medio do campo de pressao:\s*([\d\.eE\-]+)", texto)
    mse_campo_std = re.search(r"MSE medio do campo de pressao:.*?Desvio padrao:\s*([\d\.eE\-]+)", texto, re.DOTALL)

    mse_sensores = re.search(r"MSE medio dos sensores de pressao:\s*([\d\.eE\-]+)", texto)
    mse_sensores_std = re.search(r"MSE medio dos sensores de pressao:.*?Desvio padrao:\s*([\d\.eE\-]+)", texto, re.DOTALL)

    mse_sensores = re.search(r"MSE medio dos sensores de pressao:\s*([\d\.eE\-]+)", texto)

    energia_sensor = re.search(r"Energia refletida \(sensor\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", texto)
    energia_field = re.search(r"Energia ultimo frame:\s*([\d\.eE\-\+]+)", texto)

    return {
        "simulador": simulador.group(1).strip() if simulador else 0,
        "tempo_medio": float(tempo_medio.group(1)) if tempo_medio else 0,
        "tempo_std": float(tempo_std.group(1)) if tempo_std else 0,
        "mse_campo": float(mse_campo.group(1)) if mse_campo else 0,
        "mse_campo_std": float(mse_campo_std.group(1)) if mse_campo_std else 0,
        "mse_sensores": float(mse_sensores.group(1)) if mse_sensores else 0,
        "mse_sensores_std": float(mse_sensores_std.group(1)) if mse_sensores_std else 0,
        "energia_sensor": float(energia_sensor.group(1)) if energia_sensor else 0,
        "energia_field": float(energia_field.group(1)) if energia_field else 0
    }


dados = [extrair_dados(arq) for arq in arquivos_txt]
df = pd.DataFrame(dados)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

df_sorted = df.sort_values("tempo_medio")
axs[0].bar(df_sorted['simulador'], df_sorted['tempo_medio'], yerr=df_sorted['tempo_std'], capsize=5)
axs[0].set_title("Tempo médio de total (s)")
axs[0].set_ylabel("Tempo (segundos)")
axs[0].set_xticklabels(df_sorted['simulador'], rotation=45, ha='right')

df_sorted = df.sort_values("mse_campo")
axs[1].bar(df_sorted['simulador'], df_sorted['mse_campo'], yerr=df_sorted['mse_campo_std'], capsize=5)
axs[1].set_title("MSE médio do campo de pressão")
axs[1].set_yscale('log')
axs[1].set_xticklabels(df_sorted['simulador'], rotation=45, ha='right')

df_sorted = df.sort_values("mse_sensores")
axs[2].bar(df_sorted['simulador'], df_sorted['mse_sensores'], yerr=df_sorted['mse_sensores_std'], capsize=5)
axs[2].set_title("MSE médio dos sensores de pressão")
axs[2].set_yscale('log')
axs[2].set_xticklabels(df_sorted['simulador'], rotation=45, ha='right')

fig, axs = plt.subplots(1, 2, figsize=(18, 6))
df_sorted = df.sort_values("energia_sensor")
axs[0].bar(df_sorted['simulador'], df_sorted['energia_sensor'], capsize=5)
axs[0].set_title("Energia refletida no sensor")
axs[0].set_xticklabels(df_sorted['simulador'], rotation=45, ha='right')

df_sorted = df.sort_values("energia_field")
axs[1].bar(df_sorted['simulador'], df_sorted['energia_field'], capsize=5)
axs[1].set_title("Energia no ultimo frame salvo")
axs[1].set_xticklabels(df_sorted['simulador'], rotation=45, ha='right')


plt.tight_layout()
plt.show()
