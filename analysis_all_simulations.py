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
    tempo_medio = re.search(r"Tempo medio de execucao:\s*([\d\.]+)s", texto)
    tempo_std = re.search(r"Tempo medio de execucao:.*?Desvio padrao:\s*([\d\.eE\-]+)", texto, re.DOTALL)

    mse_campo = re.search(r"MSE medio do campo de pressao:\s*([\d\.eE\-]+)", texto)
    mse_campo_std = re.search(r"MSE medio do campo de pressao:.*?Desvio padrao:\s*([\d\.eE\-]+)", texto, re.DOTALL)

    mse_sensores = re.search(r"MSE medio dos sensores de pressao:\s*([\d\.eE\-]+)", texto)
    mse_sensores_std = re.search(r"MSE medio dos sensores de pressao:.*?Desvio padrao:\s*([\d\.eE\-]+)", texto, re.DOTALL)

    return {
        "simulador": simulador.group(1).strip() if simulador else None,
        "tempo_medio": float(tempo_medio.group(1)) if tempo_medio else None,
        "tempo_std": float(tempo_std.group(1)) if tempo_std else None,
        "mse_campo": float(mse_campo.group(1)) if mse_campo else None,
        "mse_campo_std": float(mse_campo_std.group(1)) if mse_campo_std else None,
        "mse_sensores": float(mse_sensores.group(1)) if mse_sensores else None,
        "mse_sensores_std": float(mse_sensores_std.group(1)) if mse_sensores_std else None,
    }


dados = [extrair_dados(arq) for arq in arquivos_txt]
df = pd.DataFrame(dados)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

df_sorted = df.sort_values("tempo_medio")
axs[0].bar(df_sorted['simulador'], df_sorted['tempo_medio'], yerr=df_sorted['tempo_std'], capsize=5)
axs[0].set_title("Tempo médio de execução (s)")
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

plt.tight_layout()
plt.show()
