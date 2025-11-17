# =======================
# Importacao de pacotes de uso geral
# =======================
import numpy as np

#Futuramente colocar aqui as preparacoes para atenuacao, por enquanto sua estrutura é provisoria para apenas compilar o codigo com essa nova estrutura

class AttenuationCoefficients:

    def __init__(self):

        self._n_sls = 3

        self._kappa_coeffs = (
            np.array([0.024081581857536852, 0.0046996089908613505, 0.00095679978724359251], dtype=np.float32),# * 0.48 ,
            np.array([0.022560146386368083, 0.0045084712797122525, 0.00089378764037688395], dtype=np.float32) #* 0.48
        )

        self._non_kappa_coeffs = (
            np.array([0.024305444805272164, 0.0047281078292263955, 0.00096672526958635019], dtype=np.float32),# * 0.48,
            np.array([0.022509197794294895, 0.0045013880073380965, 0.00089173320953691182], dtype=np.float32)# * 0.48
        )

        self._tau_epsilon_p, self._tau_sigma_p = self._kappa_coeffs
        self._tau_epsilon_s, self._tau_sigma_s = self._non_kappa_coeffs

        alpha_p = self._tau_epsilon_p / self._tau_sigma_p
        alpha_s = self._tau_epsilon_s / self._tau_sigma_s
        self._sum_alpha_p = sum(alpha_p)
        self._sum_alpha_s = sum(alpha_s)
