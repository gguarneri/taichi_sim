import numpy as np

# ----------------------------------------------------------
# Definicao de tipos usuais
# ----------------------------------------------------------
flt32 = np.float32
int32 = np.int32

# ----------------------------------------------------------
# Definicao de constantes
# ----------------------------------------------------------
# Constantes
PI = flt32(np.pi)
DEGREES_TO_RADIANS = flt32(PI / 180.0)
ZERO = flt32(0.0)
STABILITY_THRESHOLD = flt32(1.0e38)  # Limite para considerar que a simulacao esta instavel
HUGEVAL = flt32(1.0e30)  # Valor enorme

# Definicao das constantes para a o calculo das derivadas, seguindo Lui 2009 (10.1111/j.1365-246X.2009.04305.x)
coefs_Lui = [
    [9.0 / 8.0, -1.0 / 24.0],
    [75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0],
    [1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0],
    [19845.0 / 16384.0, -735.0 / 8192.0, 567.0 / 40960.0, -405.0 / 229376.0, 35.0 / 294912.0],
    [160083.0 / 131072.0, -12705.0 / 131072.0, 22869.0 / 1310720.0, -5445.0 / 1835008.0, 847.0 / 2359296.0,
     -63.0 / 2883584.0]
]


# -------------------------------
# Definicao de excecoes do modulo
# -------------------------------
class CourantError(ValueError):
    """_summary_Exceção disparada quando o número de Courant está superior a 1.0.
    Isso causa instabilidade na simulação.

    Args:
        ValueError (mensagem): Texto explicativo da exceção.
        
        ValueError (valor): Valor do número de Courant.
    """
    def __init__(self, mensagem, valor):
        super().__init__(mensagem)
        self.valor = valor

    def __str__(self):
        return f"{super().__str__()} [Valor: {self.valor}]"


class StabilityError(ValueError):
    """Exceção disparada quando a simulação está se tornando instável.

    Args:
        ValueError (_type_): _description_
    """
    def __init__(self, mensagem, valor):
        super().__init__(mensagem)
        self.valor = valor

    def __str__(self):
        return f"{super().__str__()} [Valor: {self.valor}]"


# Simbolos exportaveis do modulo
__all__ = ['flt32',
           'int32',
           'PI',
           'DEGREES_TO_RADIANS',
           'ZERO',
           'STABILITY_THRESHOLD',
           'HUGEVAL',
           'coefs_Lui', 
           'CourantError', 
           'StabilityError']
