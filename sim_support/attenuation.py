# =======================
# Importacao de pacotes de uso geral
# =======================
import numpy as np

#Futuramente colocar aqui as preparacoes para atenuacao, por enquanto sua estrutura é provisoria para apenas compilar o codigo com essa nova estrutura

class AttenuationCoefficients:

    def __init__(self):

        self._n_sls = 3

        self._kappa_coeffs = (
            np.array([1.1063680886151313E-007,3.1856366036932041E-008,9.2197868528094345E-009], dtype=np.float32),# * 0.48 ,
            np.array([1.1026580338785149E-007,3.1830988624149110E-008,9.1888147513272412E-009], dtype=np.float32) #* 0.48
        )

        self._non_kappa_coeffs = (
            np.array([1.6007136049523215E-007,3.1849176860846836E-008,6.3419851645615403E-009], dtype=np.float32),# * 0.48,
            np.array([1.5988027842097575E-007,3.1825778973240451E-008,6.3344086277835006E-009], dtype=np.float32)# * 0.48
        )

        self._tau_epsilon_p, self._tau_sigma_p = self._kappa_coeffs
        self._tau_epsilon_s, self._tau_sigma_s = self._non_kappa_coeffs

        alpha_p = self._tau_epsilon_p / self._tau_sigma_p
        alpha_s = self._tau_epsilon_s / self._tau_sigma_s
        self._sum_alpha_p = sum(alpha_p)
        self._sum_alpha_s = sum(alpha_s)