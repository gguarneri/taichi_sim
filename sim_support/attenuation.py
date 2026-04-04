# =======================
# Importacao de pacotes de uso geral
# =======================
import numpy as np

#Futuramente colocar aqui as preparacoes para atenuacao, por enquanto sua estrutura é provisoria para apenas compilar o codigo com essa nova estrutura

class AttenuationCoefficients:

    def __init__(self):

        self._n_sls = 3

        self._kappa_coeffs = (
            np.array([2.3183195804298614e-002, 4.5879690742179936e-003, 9.1870040362984834e-004], dtype=np.float32),# * 0.48 ,
            np.array([2.2767696890764546e-002, 4.5367506354972902e-003, 9.0203872869258736e-004], dtype=np.float32) #* 0.48
        )

        self._non_kappa_coeffs = (
            np.array([2.3851313121087559e-002, 4.6706068163414900e-003, 9.4677570922160968e-004], dtype=np.float32),# * 0.48,
            np.array([2.2613794698025876e-002, 4.5158937361154533e-003, 8.9594510015710547e-004], dtype=np.float32)# * 0.48
        )

        self._tau_epsilon_p, self._tau_sigma_p = self._kappa_coeffs
        self._tau_epsilon_s, self._tau_sigma_s = self._non_kappa_coeffs

        alpha_p = self._tau_epsilon_p / self._tau_sigma_p
        alpha_s = self._tau_epsilon_s / self._tau_sigma_s
        self._sum_alpha_p = sum(alpha_p)
        self._sum_alpha_s = sum(alpha_s)


  