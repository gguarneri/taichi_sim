# =======================
# Importacao de pacotes de uso geral
# =======================
import numpy as np

class Field:
    def __init__(self, dim):
        self._dim = dim
        self._data = np.zeros(self._dim, dtype=np.float32)
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        if new_data.shape == self._data.shape:
            self._data = new_data
        else:
            raise ValueError