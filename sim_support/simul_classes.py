"""_summary_

Raises:
    TypeError: _description_
    IndexError: _description_
    IndexError: _description_
    IndexError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""
from sim_support import *
import numpy as np
from scipy.signal import gausspulse
from sim_support.gauss_wavelets import (gaussian_pulse,
                                        gaussian_first_dev_pulse,
                                        gaussian_second_dev_pulse)


class SimulationROI:
    """
    Classe que armazena os parâmetros da *Region of Interest* (ROI) para
    a simulação.


    Parameters
    ----------
        coord_ref : :class:`np.ndarray`
            Ponto cartesiano indicando a coordenada de referência da ROI, em
            mm. É referente ao ponto da ROI com índices (0, 0, 0), sem considerar as camadas e CPML.
            Por padrão, é (0.0, 0.0, 0.0) mm.

        height : int, float
            Altura da ROI, em mm. Corresponde ao eixo `z`. Por padrão, é 30.0 mm.

        h_len : int
            Quantidade de pontos na dimensão de altura ROI. Por padrão, é 300.

        width : int, float
            Largura da ROI (em um transdutor linear, tipicamente corresponde à
            direção ativa -- eixo `x`), em mm. Por padrão, é 30.0 mm.

        w_len : int
            Quantidade de pontos na dimensão de largura ROI. Por padrão, é 300.

        depth : int, float
            Profundidade da ROI (em um transdutor linear, tipicamente corresponde à
            direção passiva -- eixo `y`), em mm. Por padrão, é 0.0 mm (ROI de duas dimensões).

        d_len : int
            Quantidade de pontos na dimensão de profundidade ROI. Por padrão, é 1
            (ROI de duas dimensões).

        pad : int
            Quantidade de pontos adicionais em cada lado da dimensao da ROI. Por padrão é 1.

    Attributes
    ----------
        coord_ref : :class:`np.ndarray`
            Ponto cartesiano indicando a coordenada de referência da ROI, em mm.

        h_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da altura (dimensão 3 -- eixo `z`)
            da ROI de simulação, em mm.

        _h_len : int
            Quantidade de pontos da ROI no sentido da altura.

        h_step : float
            Tamanho do passo dos pontos da ROI no sentido da altura, em mm.

        height : float
            Altura da ROI, em mm.

        w_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da largura (dimensão 1 -- eixo `x`)
            da ROI de simulação.

        _w_len : int
            Quantidade de pontos da ROI no sentido da largura.

        w_step : float
            Tamanho do passo dos pontos da ROI no sentido da largura, em mm.

        width : float
            Largura da ROI, em mm.

        d_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 2 -- eixo `y`)
            da ROI de simulação.

        _d_len : int
            Quantidade de pontos da ROI no sentido da profundidade.

        d_step : float
            Tamanho do passo dos pontos da ROI no sentido da profundidade, em mm.

        depth : float
            Profundidade da ROI, em mm.

        _pad : int
            Quantidade adicional de pontos em cada lado das dimensões da ROI.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se ``coord_ref`` não for do tipo
        :class:`np.ndarray` e/ou não possuir 1 linha e três colunas.

    Notes
    -----
    Esta classe aplica-se a ROIs em duas e três dimensões.

    """

    def __init__(self, coord_ref=np.zeros((1, 3)), height=30.0, h_len=300, width=30.0, w_len=300, depth=0.0, d_len=1,
                 len_pml_xmin=10, len_pml_xmax=10, len_pml_ymin=10, len_pml_ymax=10, len_pml_zmin=10, len_pml_zmax=10,
                 rcoef_pml=0.0001, npower_pml=2.0, k_max_pml=1.0,
                 pad=1, rho_map=None):
        if type(coord_ref) is list:
            coord_ref = np.array(coord_ref, dtype=np.float32)

        if (coord_ref is not np.ndarray) and (coord_ref.shape != (1, 3)):
            raise TypeError("``coord_ref`` deve ser um vetor-linha de 3 elementos [shape = (1,3)]")

        # Atribuicao dos atributos da instancia.
        # Ponto cartesiano indicando a coordenada de referencia da ROI.
        self.coord_ref = coord_ref

        # Passo dos pontos da ROI no sentido da altura.
        self.h_step = height / h_len
        self._dec_h = int(abs(np.log10(self.h_step))) + 2
        self.h_step = np.round(self.h_step, decimals=self._dec_h).astype(np.float32)

        # Passo dos pontos da ROI no sentido da largura.
        self.w_step = width / w_len
        self._dec_w = int(abs(np.log10(self.w_step))) + 2
        self.w_step = np.round(self.w_step, decimals=self._dec_w).astype(np.float32)

        # Passo dos pontos da ROI no sentido da profundidade.
        if depth > 0.0 and d_len > 1:
            self.d_step = depth / d_len
        else:
            self.d_step = self.w_step
        self._dec_d = int(abs(np.log10(self.d_step))) + 2
        self.d_step = np.round(self.d_step, decimals=self._dec_d).astype(np.float32)

        # Se for definido um mapa de densidades, pega o numero de pontos do mapa
        w_len_map = 0
        h_len_map = 0
        d_len_map = 0
        if rho_map is not None:
            if len(rho_map.shape) == 1:
                w_len_map = rho_map.shape[0]
            elif len(rho_map.shape) == 2:
                w_len_map, h_len_map = rho_map.shape
            elif len(rho_map.shape) == 3:
                w_len_map, d_len_map, h_len_map = rho_map.shape

        # Ajusta tamanhos e passos no caso de definicao de mapa de densidades
        if w_len_map > w_len:
            self.width = np.float32(w_len_map * self.w_step)
            self._w_len = w_len_map
        else:
            self.width = np.float32(width)
            self._w_len = w_len

        if h_len_map > h_len:
            self.height = np.float32(h_len_map * self.h_step)
            self._h_len = h_len_map
        else:
            self.height = np.float32(height)
            self._h_len = h_len

        if d_len_map > d_len:
            self.depth = np.float32(d_len_map * self.d_step)
            self._d_len = d_len_map
        else:
            self.depth = np.float32(depth)
            self._d_len = d_len

        # Vetor com as coordenadas da ROI no sentido da largura (dimensão 1 - eixo 'x') da simulacao.
        self.w_points = np.linspace(coord_ref[0, 0], coord_ref[0, 0] + self.width, num=int(self._w_len),
                                    endpoint=False, dtype=np.float32).round(decimals=self._dec_w)

        # Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 2 - eixo 'y') da simulacao.
        self.d_points = np.linspace(coord_ref[0, 1], coord_ref[0, 1] + self.depth, num=int(self._d_len),
                                    endpoint=False, dtype=np.float32).round(decimals=self._dec_d)

        # Vetor com as coordenadas da ROI no sentido da altura (dimensao 3 - eixo 'z') da simulacao.
        self.h_points = np.linspace(coord_ref[0, 2], coord_ref[0, 2] + self.height, num=int(self._h_len),
                                    endpoint=False, dtype=np.float32).round(decimals=self._dec_h)

        # Tamanho das camadas de PML, 0 se não for para calcular
        self._pml_xmin_len = len_pml_xmin
        self._pml_xmax_len = len_pml_xmax
        self._pml_ymin_len = len_pml_ymin
        self._pml_ymax_len = len_pml_ymax
        self._pml_zmin_len = len_pml_zmin
        self._pml_zmax_len = len_pml_zmax
        
        # Parametros da PML
        self._rcoef_pml = rcoef_pml
        self._npower_pml = npower_pml
        self._k_max_pml = k_max_pml

        # Quantidade adicional de pontos em cada lado da ROI.
        self._pad = pad

    def has_pml(self):
        return (self._pml_xmin_len,
                self._pml_xmax_len,
                self._pml_ymin_len,
                self._pml_ymax_len,
                self._pml_zmin_len,
                self._pml_zmax_len) != (0, 0, 0, 0, 0, 0)
    
    def get_dx(self):
        return flt32(self.w_step)
    
    def get_dy(self):
        return flt32(self.d_step)
    
    def get_dz(self):
        return flt32(self.h_step)
    
    def get_nx(self):
        return self._w_len + self._pml_xmin_len + self._pml_xmax_len + 2 * self._pad

    def get_ny(self):
        return self._d_len + self._pml_ymin_len + self._pml_ymax_len + 2 * self._pad

    def get_nz(self):
        return self._h_len + self._pml_zmin_len + self._pml_zmax_len + 2 * self._pad

    def get_len_x(self):
        return self._w_len

    def get_len_y(self):
        return self._d_len

    def get_len_z(self):
        return self._h_len

    def get_ix_min(self):
        return self._pml_xmin_len + self._pad

    def get_ix_max(self):
        return self._w_len + self._pml_xmin_len + self._pad

    def get_iy_min(self):
        return self._pml_ymin_len + self._pad

    def get_iy_max(self):
        return self._d_len + self._pml_ymin_len + self._pad

    def get_iz_min(self):
        return self._pml_zmin_len + self._pad

    def get_iz_max(self):
        return self._h_len + self._pml_zmin_len + self._pad

    def get_pml_thickness_x(self):
        return (self._pml_xmin_len + self._pml_xmax_len) * self.w_step

    def get_pml_thickness_y(self):
        return (self._pml_ymin_len + self._pml_ymax_len) * self.d_step

    def get_pml_thickness_z(self):
        return (self._pml_zmin_len + self._pml_zmax_len) * self.h_step
    
    def get_dec(self):
        return self._dec_w, self._dec_d, self._dec_h

    def is_point_in_roi(self, point):
        """
        Função para retornar se o ponto pertence a ROI.
        """
        if type(point) is not np.ndarray and type(point) is list:
            point = np.array(point, dtype=np.float32)
        elif type(point) is np.array:
            point = point.astype(np.float32)

        if (not (self.w_points[0] <= point[0] <= self.w_points[-1]) or
                not (self.d_points[0] <= point[1] <= self.d_points[-1]) or
                not (self.h_points[0] <= point[2] <= self.h_points[-1])):
            return False
        else:
            return True

    def get_nearest_grid_idx(self, point):
        """
        Função para retornar os índices mais próximos da grade para o ponto da ROI fornecido.
        """
        if type(point) is not np.ndarray and type(point) is list:
            point = np.array(point, dtype=np.float32)
        elif type(point) is np.array:
            point = point.astype(np.float32)

        if not self.is_point_in_roi(point):
            raise IndexError(f"[{point[0]}, {point[1]}, {point[2]}] out of bounds")

        ix = np.absolute(self.w_points - np.round(point[0] - self.w_step / 10.0 ** (self._dec_w - 1),
                                                  decimals=self._dec_w)).argmin() + self._pml_xmin_len + self._pad
        iy = np.absolute(self.d_points - np.round(point[1] - self.d_step / 10.0 ** (self._dec_d - 1),
                                                  decimals=self._dec_d)).argmin() + self._pml_ymin_len + self._pad
        iz = np.absolute(self.h_points - np.round(point[2] - self.h_step / 10.0 ** (self._dec_h - 1),
                                                  decimals=self._dec_h)).argmin() + self._pml_zmin_len + self._pad
        return [ix, iy, iz]

    def calc_pml_array(self, axis='x', grid='f', dt=1.0, cp=1.0, alpha_max=1.0):
        """
        Função que calcula os vetores com os valores para implementar a camada de PML.
        """
        # Origem da PML (posicao das bordas direita e esquerda menos a espessura, em unidades de distancia)
        if axis == 'x' or axis == 'X':
            delta = np.float32(self.w_step)
            tam_pml = self._w_len + self._pml_xmin_len + self._pml_xmax_len
            orig_left = self._pml_xmin_len * delta
            orig_right = (self._pml_xmin_len + self._w_len - 1) * delta
            thickness_pml_left = self._pml_xmin_len * delta
            thickness_pml_right = self._pml_xmax_len * delta
            dec = self._dec_w
            d0 = flt32(-(self._npower_pml + 1.0) * cp * np.log(self._rcoef_pml) / self.get_pml_thickness_x())
        elif axis == 'y' or axis == 'Y':
            delta = np.float32(self.d_step)
            tam_pml = self._d_len + self._pml_ymin_len + self._pml_ymax_len
            orig_left = self._pml_ymin_len * delta
            orig_right = (self._pml_ymin_len + self._d_len - 1) * delta
            thickness_pml_left = self._pml_ymin_len * delta
            thickness_pml_right = self._pml_ymax_len * delta
            dec = self._dec_d
            d0 = flt32(-(self._npower_pml + 1.0) * cp * np.log(self._rcoef_pml) / self.get_pml_thickness_y())
        elif axis == 'z' or axis == 'Z':
            delta = np.float32(self.h_step)
            tam_pml = self._h_len + self._pml_zmin_len + self._pml_zmax_len
            orig_left = self._pml_zmin_len * delta
            orig_right = (self._pml_zmin_len + self._h_len - 1) * delta
            thickness_pml_left = self._pml_zmin_len * delta
            thickness_pml_right = self._pml_zmax_len * delta
            dec = self._dec_h
            d0 = flt32(-(self._npower_pml + 1.0) * cp * np.log(self._rcoef_pml) / self.get_pml_thickness_z())
        else:
            raise IndexError(f"'axis' = {axis} not supported")

        # Inicializacao para full ou half grid
        val = np.round(delta * np.arange(tam_pml), decimals=dec).astype(np.float32)
        if grid == 'f' or grid == 'F':
            val_pml_left = np.round(orig_left - val, decimals=dec)
            val_pml_right = np.round(val - orig_right, decimals=dec)
        elif grid == 'h' or grid == 'H':
            val_pml_left = np.round(orig_left - (val + delta / 2.0), decimals=dec).astype(np.float32)
            val_pml_right = np.round((val + delta / 2.0) - orig_right, decimals=dec).astype(np.float32)
        else:
            raise IndexError(f"'grid' = {grid} not supported")

        # Calculo dos coeficientes
        pml_mask_left = np.where(val_pml_left < 0.0, False, True) if thickness_pml_left > 0.0 \
            else np.zeros(val_pml_left.shape, dtype=bool)
        pml_mask_right = np.where(val_pml_right < 0.0, False, True) if thickness_pml_right > 0.0 \
            else np.zeros(val_pml_right.shape, dtype=bool)
        mask = np.logical_or(pml_mask_left, pml_mask_right)
        pml = np.zeros(tam_pml, dtype=np.float32)
        if thickness_pml_left:
            pml[pml_mask_left] = val_pml_left[pml_mask_left] / thickness_pml_left
        if thickness_pml_right:
            pml[pml_mask_right] = val_pml_right[pml_mask_right] / thickness_pml_right
        d = (d0 * pml ** self._npower_pml).astype(np.float32)
        k = (1.0 + (self._k_max_pml - 1.0) * pml ** self._npower_pml).astype(np.float32)
        alpha = (alpha_max * (1.0 - np.where(mask, pml, 1.0))).astype(np.float32)
        b = np.exp(-(d / k + alpha) * dt).astype(np.float32)
        a = np.zeros(tam_pml, dtype=np.float32)
        i = np.where(d > 1e-6)
        a[i] = d[i] * (b[i] - 1.0) / (k[i] * (d[i] + k[i] * alpha[i]))

        return a, b, k


class SimulationProbe:
    """
    Classe base contendo as configurações do transdutor para a simulação.

    Na implementação atual, os tipos suportados são mono e linear.

    Parameters
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa ao centro geométrico do transdutor. Por padrão se assume os
            valores [0.0, 0.0, 0.0].

    Attributes
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa ao centro geométrico do transdutor.

    """

    def __init__(self, coord_center=np.zeros((1, 3)), dec=(2, 2, 2)):
        # Coordenada central do transdutor, em relacao ao ponto de referencia da ROI.
        self.coord_center = (coord_center if coord_center is np.ndarray else np.array(coord_center)).astype(np.float32)
        
        # Casas decimais para arredondamentos
        self._dec = dec


class ElementRect:
    """
    Classe que define um elemento retangular de um transdutor de ultrassom.
    Essa classe pode ser utilizada nos transdutores "LinearArray" e "MonoRect".
    """

    def __init__(self, dim_a=0.5, dim_p=10.0, coord_center=np.zeros((1, 3)),
                 freq=5., bw=0.5, gain=1.0, t0=1.0,
                 tx_en=True, rx_en=True, pulse_type="gaussian"):
        # Dimensao no sentido ativo do transdutor.
        self.elem_dim_a = np.float32(dim_a)

        # Dimensao no sentido passivo do transdutor.
        self.elem_dim_p = np.float32(dim_p)

        # Coordenada central do elemento
        self.coord_center = coord_center.astype(np.float32)

        # Parametros referentes ao sinal de excitacao do transdutor.
        # Frequencia central, em MHz.
        self.freq = np.float32(freq)

        # Banda passante, em percentual da frequencia central.
        self.bw = np.float32(bw)

        # Ganho do elemento transdutor.
        self.gain = np.float32(gain)

        # Atraso do sinal de excitacao.
        self.t0 = np.float32(t0)

        # Flag se e emissor.
        self.tx_en = tx_en

        # Flag se e receptor.
        self.rx_en = rx_en

        # Tipo do pulso de excitacao. O unico tipo possivel e: ``gaussian``.
        self.pulse_type = pulse_type

    def get_element_exc_fn(self, t, out='r'):
        dt = t[1] - t[0]
        gp, _, egp = gausspulse((t - self.t0), fc=self.freq, bw=self.bw, retquad=True, retenv=True)
        eps = np.finfo(flt32).eps
        if out == 'e':
            egp[np.abs(egp) < eps] = 0.0
            ss = flt32(egp)
        else:
            gp[np.abs(gp) < eps] = 0.0
            ss = flt32(gp)

        return np.diff(self.gain * np.float32(ss) / dt, append=0.0).astype(flt32)

    def get_num_points_roi(self, sim_roi=SimulationROI(), simul_type="2D"):
        """
        Função que retorna o número dos pontos ativos do transdutor no grid de simulação.

        Returns
        -------
            : int
                Quantidade de pontos ativos (fontes) do elemento transdutor.

        """
        dec_w, dec_d, dec_h = sim_roi.get_dec()
        dim_p = min(self.elem_dim_p, sim_roi.depth)
        num_pt_a = int(np.round(self.elem_dim_a / sim_roi.w_step, decimals=dec_w))
        num_pt_p = int(np.round(dim_p / sim_roi.d_step, decimals=dec_d)) if dim_p != 0.0 else 1
        simul_type = simul_type.lower()
        num_coord = num_pt_a
        if simul_type == "3d":
            num_coord *= num_pt_p

        return num_coord

    def get_points_roi(self, sim_roi=SimulationROI(), probe_center=np.zeros((1, 3)), simul_type="2D", dir="e"):
        """
        Função que retorna as coordenadas de todos os pontos ativos do transdutor no grid de simulação,
        no formato vetorizado.

        Returns
        -------
            : :class:`np.ndarray`
                Matriz :math:`M` x 3, em que :math:`M` é a quantidade de
                pontos ativos (fontes) do elemento transdutor como índices de pontos na ROI.
                Cada linha dessa matriz e o indice 3D de um ponto na ROI.

        """
        if type(dir) is str:
            if (dir.lower() == "e" and not self.tx_en) or (dir.lower() == "r" and not self.rx_en):
                return list()
        else:
            raise ValueError("'dir' must be a string")

        dec_w, dec_d, dec_h = sim_roi.get_dec()
        dim_p = min(self.elem_dim_p, sim_roi.depth)
        num_pt_a = int(np.round(self.elem_dim_a / sim_roi.w_step, decimals=dec_w) + 0.5)
        num_pt_p = int(np.round(dim_p / sim_roi.d_step, decimals=dec_d) + 0.5) if dim_p != 0.0 else 1
        num_coord = num_pt_a
        if simul_type.lower() == "3d":
            num_coord *= num_pt_p

        # Calcula a coordenada do primeiro ponto
        x_coord = np.float32(self.coord_center[0] - ((self.elem_dim_a - sim_roi.w_step) / 2.0 if num_pt_a // 2 else
                                                     (self.elem_dim_a / 2.0)))
        y_coord = np.float32(0.0 if simul_type.lower() == "2d" else
                             (self.coord_center[1] - ((dim_p - sim_roi.d_step) / 2.0) if num_pt_p // 2 else
                              (dim_p / 2.0)))
        z_coord = self.coord_center[2]

        # Pega os indices na ROI da coordenada do primeiro ponto
        point_coord = np.array([x_coord, y_coord, z_coord], np.float32) + probe_center.astype(np.float32)
        point_0 = sim_roi.get_nearest_grid_idx(point_coord)

        # Monta lista de pontos
        list_out = [ [point_0[0] + (p % num_pt_a),
                      point_0[1] + 0 if simul_type == "2d" else ((p // num_pt_a) % num_pt_p),
                      point_0[2]] for p in range(num_coord)]

        return list_out

class ElementCirc:
    """
    Classe que define um elemento circular de um transdutor de ultrassom.
    Essa classe pode ser utilizada no transdutor "MonoCirc" até o momento.
    """

    def __init__(self, radius=0.25, coord_center=np.zeros((3,)),
                 freq=5., bw=0.5, gain=1.0, t0=1.0,
                 tx_en=True, rx_en=True, pulse_type="gaussian"):
        
        # Dimensões e Posição
        self.radius = np.float32(radius)
        
        # Coordenada central do elemento
        self.coord_center = coord_center.astype(np.float32)

        # Parâmetros de Sinal
        self.freq = np.float32(freq)
        self.bw = np.float32(bw)
        self.gain = np.float32(gain)
        self.t0 = np.float32(t0)
        self.tx_en = tx_en
        self.rx_en = rx_en
        self.pulse_type = pulse_type

    def get_element_exc_fn(self, t, out='r'):
        dt = t[1] - t[0]
        gp, _, egp = gausspulse((t - self.t0), fc=self.freq, bw=self.bw, retquad=True, retenv=True)
        eps = np.finfo(flt32).eps
        if out == 'e':
            egp[np.abs(egp) < eps] = 0.0
            ss = flt32(egp)
        else:
            gp[np.abs(gp) < eps] = 0.0
            ss = flt32(gp)

        return np.diff(self.gain * np.float32(ss) / dt, append=0.0).astype(flt32)

    def _get_candidate_points(self, sim_roi, simul_type, current_center):
        """
        Método auxiliar que calcula pontos baseado em um centro específico.
        """
        dec_w, dec_d, dec_h = sim_roi.get_dec()
        simul_type = simul_type.lower()

        num_pt_a = int(np.round((2.0 * self.radius) / sim_roi.w_step, decimals=dec_w))
        
        if simul_type == "3d":
            depth_radius = min(self.radius, sim_roi.depth)
            num_pt_p = int(np.round((2.0 * depth_radius) / sim_roi.d_step, decimals=dec_d))
        else:
            num_pt_p = 1
            depth_radius = 0.0

        x_start = np.float32(current_center[0] - self.radius)
        
        y_start = np.float32(0.0 if simul_type == "2d" else current_center[1] - depth_radius)
        z_coord = current_center[2]

        corner_coord = np.array([x_start, y_start, z_coord], dtype=flt32)
        corner_idx = sim_roi.get_nearest_grid_idx(corner_coord)

        list_out = list()
        
        for jj in range(num_pt_a):
            dx = (jj - (num_pt_a - 1) / 2.0) * sim_roi.w_step
            
            for zz in range(num_pt_p):
                if simul_type == "3d":
                    dy = (zz - (num_pt_p - 1) / 2.0) * sim_roi.d_step
                else:
                    dy = 0.0

                # Verifica se está dentro do raio (equação do círculo)
                if np.sqrt(dx**2 + dy**2) <= self.radius:
                    ix = corner_idx[0] + jj
                    # Em 2D, mantemos o índice Y original; em 3D iteramos ip
                    iy = corner_idx[1] + (0 if simul_type == "2d" else zz)
                    iz = corner_idx[2]
                    
                    list_out.append([ix, iy, iz])

        return list_out

    def get_num_points_roi(self, sim_roi, simul_type="2D"):
        """
        Função que retorna o número dos pontos ativos do transdutor no grid de simulação.
        
        """
        
        return len(self._get_candidate_points(sim_roi, simul_type, self.coord_center))

    def get_points_roi(self, sim_roi, probe_center=np.zeros((3,)), simul_type="2D", dir="e"):
        """
        Função que retorna as coordenadas de todos os pontos ativos do transdutor no grid de simulação,
        no formato vetorizado.
        
        """
        if not isinstance(dir, str):
            raise ValueError("'dir' must be a string")
            
        if (dir.lower() == "e" and not self.tx_en) or (dir.lower() == "r" and not self.rx_en):
            return list()

        # Calcula o centro efetivo somando o offset do probe
        # O flatten garante consistência dimensional
        probe_offset = probe_center.astype(flt32).flatten()
        effective_center = self.coord_center + probe_offset

        # Passa o centro calculado sem alterar o estado do objeto
        return self._get_candidate_points(sim_roi, simul_type, effective_center)
    

class SimulationProbeLinearArray(SimulationProbe):
    """
    Classe contendo as configurações de um transdutor do tipo array linear.
    É uma classe derivada de ``SimulationProbe``, específica para os transdutore do tipo
    "LinearArray".

    Parameters
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa ao centro geométrico do transdutor. Por padrão se assume os
            valores [0.0, 0.0, 0.0].

        num_elem : int
            Número de elementos. Exclusivo para transdutores do tipo
            ``linear``. Por padrão, é 32.

        dim_a : int, float
            Dimensão na direção ativa elementos do transdutor, em mm. Por padrão é 0.5 mm.

        inter_elem : int, float
            Espaçamento entre elementos, em mm. Por padrão é 0.1 mm.

        freq : int, float
            Frequência central, em MHz. Por padrão, é 5 MHz.

        bw : int, float
            Banda passante, em percentual da frequência central. Por padrão,
            é 0.5 (50%).

        pulse_type : str
            Tipo do pulso de excitação. Os tipos possíveis são: ``gaussian``,
            ``cossquare``, ``hanning`` e ``hamming``. Por padrão, é
            ``gaussian``.

    Attributes
    ----------
        num_elem : int
            Número de elementos. Exclusivo para transdutores da classe *array*.

        inter_elem : int, float
            Espaçamento entre elementos, em mm. Exclusivo para transdutores da
            classe *array*.

        pitch: int, float
            Espaçamento entre os centros dos elementos, em mm. Exclusivo para transdutores da
            classe *array*.

        elem_list : :class:`ElementRect`
            É uma lista de objetos do tipo ``ElementRect``, contendo as caracteristicas
            físicas e elétricas dos elementos ativos do transdutor.

    """

    def __init__(self, coord_center=np.zeros((1, 3)), dec=(2, 2, 2), num_elem=32, dim_a=0.5, dim_p=10.0, inter_elem=0.1,
                 freq=5., bw=0.5, gain=1.0, pulse_type="gaussian", id="",
                 emitters="all", receivers="all", t0_emission=None, t0_reception=None):
        # Chama o construtor da classe base.
        super().__init__(coord_center, dec)

        # Identificacao do tranasdutor
        self.id = id

        # Espacamento entre elementos.
        self.inter_elem = np.float32(inter_elem)

        # Le a configuracao dos elementos emissores do transdutor
        if type(emitters) is str and emitters == "all":
            self.emitters = [True for _ in range(num_elem)]
        elif type(emitters) is str and emitters == "none":
            self.emitters = [False for _ in range(num_elem)]
        elif type(emitters) is list:
            self.emitters = [eval(el.lower().capitalize()) for el in emitters]
            if len(self.emitters) < num_elem:
                self.emitters += [False] * (num_elem - len(self.emitters))
            elif len(self.emitters) > num_elem:
                self.emitters = self.emitters[:num_elem]
        else:
            raise ValueError("emitters must be a string or a list of strings")

        # Le a configuracao dos elementos receptores do transdutor
        if type(receivers) is str and receivers == "all":
            self.receivers = [True for _ in range(num_elem)]
        elif type(receivers) is str and receivers == "none":
            self.receivers = [False for _ in range(num_elem)]
        elif type(receivers) is list:
            self.receivers = [bool(eval(el.lower().capitalize())) for el in receivers]
            if len(self.receivers) < num_elem:
                self.receivers += [False] * (num_elem - len(self.receivers))
            elif len(self.receivers) > num_elem:
                self.receivers = self.receivers[:num_elem]
        else:
            raise ValueError("receivers must be a string or a list of strings.")

        # Tempo de atraso para emissao dos elementos. Se for um valor escalar, e assumido para todos os elementos.
        # Se for um array, deve ter um valor para cada elemento.
        # Se for um nome de um arquivo 'law',
        if t0_emission is None:
            self._t0_emission = np.zeros(num_elem, dtype=np.float32)
        elif type(t0_emission) is np.float32 or type(t0_emission) is float:
            self._t0_emission = np.ones(num_elem, dtype=np.float32) * np.float32(t0_emission)
        elif type(t0_emission) is list:
            self._t0_emission = t0_emission
            if len(self._t0_emission) < num_elem:
                self._t0_emission += [0.0] * (num_elem - len(self._t0_emission))
            elif len(self._t0_emission) > num_elem:
                self._t0_emission = self._t0_emission[:num_elem]
            self._t0_emission = np.array(self._t0_emission, dtype=np.float32)
        else:
            raise ValueError("t0_emission must be either a float [numpy.float32] or a list of floats.")

        # Tempo de atraso para recepcao dos elementos. Se for um valor escalar, e assumido para todos os elementos.
        # Se for um array, deve ter um valor para cada elemento.
        if t0_reception is None:
            self._t0_reception = np.zeros(num_elem, dtype=np.float32)
        elif type(t0_reception) is np.float32 or type(t0_reception) is float:
            self._t0_reception = np.ones(num_elem, dtype=np.float32) * np.float32(t0_reception)
        elif type(t0_reception) is list:
            self._t0_reception = t0_reception
            if len(self._t0_reception) < num_elem:
                self._t0_reception += [0.0] * (num_elem - len(self._t0_reception))
            elif len(self._t0_reception) > num_elem:
                self._t0_reception = self._t0_reception[:num_elem]
            self._t0_reception = np.array(self._t0_reception, dtype=np.float32)
        else:
            raise ValueError("t0_reception must be either a float [numpy.float32] or a list of floats.")

        # Espacamento entre os centros dos elementos.
        self.pitch = np.float32(dim_a + inter_elem)

        # Numero de elementos.
        self.num_elem = num_elem
        offset_center = np.array([((num_elem - 1) * self.pitch + dim_a) / 2.0, 0.0, 0.0], dtype=np.float32)
        self.elem_list = [ElementRect(dim_a=dim_a, dim_p=dim_p,
                                      coord_center=np.round(np.array([dim_a / 2.0 + i * self.pitch, 0.0, 0.0],
                                                            dtype=np.float32) - offset_center, decimals=np.max(self._dec)),
                                      freq=freq, bw=bw, gain=gain, pulse_type=pulse_type,
                                      tx_en=self.emitters[i],
                                      rx_en=self.receivers[i],
                                      t0=np.float32(self._t0_emission[i]))
                          for i in range(num_elem)]

        # Parametros geometricos gerais do transdutor
        self._dim_a = dim_a
        self._dim_p = dim_p

        # Parametros eletricos gerais do transdutor
        self._freq = freq
        self._bw = bw
        self._gain = gain
        self._pulse_type = pulse_type

        # Transdutores matriciais
        #     # Numero de elementos.
        #     self.num_elem = num_elem
        #
        #     # Posicao central do elemento ativo do transdutor.
        #     self.elem_center = np.zeros((num_elem, 3))
        #
        #     # Espacamento entre elementos.
        #     self.inter_elem = inter_elem
        #
        # Transdutores circulares
        #     # Numero de elementos.
        #     self.num_elem = num_elem
        #
        #     # Posicao central do elemento ativo do transdutor.
        #     self.elem_center = np.zeros((num_elem, 3))
        #
        #     # Espacamento entre elementos.
        #     self.inter_elem = inter_elem
        #
        #     self.elem_list = elem_list
        #
        # Transdutores ``mono``.
        # # Formato do transdutor. Os valores possíveis sao ``circle``
        # # e ``rectangle``. O valor padrao é ``circle``.
        # self.shape = "circle"
        #
        # # Posicao central do elemento ativo do transdutor.
        # self.elem_center = np.zeros((num_elem, 3))
        #
        # # Espacamento entre elementos.
        # self.inter_elem = inter_elem
        #
        # # Espacamento entre os centros dos elementos.
        # self.pitch = dim_a + inter_elem

    def get_freq(self, mode='common'):
        """
        Função que retorna a frequência do transdutor.

        :param mode: str
            Este parâmetro define o modo de obtenção da frequência.
            "common" significa que será utilizado o parâmetro geral, utilizado por todos os elementos ativos.
            Este é o padrão.
            "mean" significa que será calculada a média das frequências de cada elemeto ativo do transdutor.
            "max" significa a maior frequência entre todos os elementos ativos do transdutor.

        :return: numpy.float32
            Retorna o valor da frequência do transdutor.
        """
        if mode == 'mean':
            return np.array([e.freq for e in self.elem_list]).mean()
        elif mode == 'max':
            return np.array([e.freq for e in self.elem_list]).max()
        else:
            return self._freq
    
    def get_points_roi(self, sim_roi=SimulationROI(), simul_type="2D", dir="e"):
        """
        Função que retorna as coordenadas de todos os pontos ativos do transdutor no grid de simulação,
        no formato vetorizado.

        Returns
        -------
            : list
                Lista com :math:`M` pontos ativos (fontes) do transdutor como índices de pontos na ROI.
                Cada elemento dessa lista é a coordenada cartesiana (como índice) de um ponto na ROI.

        """
        arr_out = list()
        idx_src = list()
        for idx_st, e in enumerate(self.elem_list):
            try:
                arr_elem = e.get_points_roi(sim_roi=sim_roi, probe_center=self.coord_center,
                                            simul_type=simul_type, dir=dir)
                arr_out += arr_elem
                if len(arr_elem):
                    idx_src += [idx_st for _ in range(len(arr_elem))]
            except IndexError:
                pass

        return arr_out, idx_src

    def get_source_term(self, samples=1000, dt=1.0, out='r'):
        """
        Retorna o sinal do termo de fonte do transdutor.
        :param out:
        :param samples: int
            Número de amostras de tempo na simulação.
        :param dt: float
            Valor do passo de tempo na simulação.

        :return: :numpy.array
        Array contém dimensões de N amostras de tempo (linhas) por M elementos do transdutor (colunas).
        """
        dec = int(abs(np.log10(dt))) + 2
        t = np.round(np.arange(samples, dtype=np.float32) * dt, decimals=dec)
        source_term = np.zeros((samples, self.num_elem), dtype=np.float32)
        for idx_st, e in enumerate(self.elem_list):
            if e.tx_en:
                source_term[:, idx_st] = e.get_element_exc_fn(t, out)

        return source_term

    def get_idx_rec(self, sim_roi=SimulationROI(), simul_type="2D"):
        """
        Função que retorna um array com o índice do receptor para cada ponto da ROI que é um ponto receptor.
        :param simul_type:
        :param sim_roi:

        :return: list
        Lista com o índice do elemento receptor de cada ponto receptor na ROI.
        """
        idx_rec = list()
        for idx_st, e in enumerate(self.elem_list):
            try:
                arr_elem = e.get_points_roi(sim_roi=sim_roi, probe_center=self.coord_center,
                                            simul_type=simul_type, dir='r')
                if len(arr_elem):
                    idx_rec += [idx_st for _ in range(len(arr_elem))]
            except IndexError:
                pass

        return idx_rec

    def get_delay_rx(self):
        """
        Função que retorna uma lista com os valores do atraso na recepção de todos os canais.

        :return: list
        Lista com o tempo de atraso de recepção, em microssegundos, de todos os canais do transdutor habilitados para
        recepção.
        """
        t0_recp = list()
        for idx_e, e in enumerate(self.elem_list):
            if e.rx_en:
                t0_recp.append(self._t0_reception[idx_e])

        return t0_recp

    def set_t0(self, t0_emission=None):
        """
        Função que modifica os valores do atraso na emissão de todos os canais.

        :return: None
        """
        if t0_emission is None:
            self._t0_emission = np.zeros(self.num_elem, dtype=np.float32)
        elif type(t0_emission) is np.float32 or type(t0_emission) is float:
            self._t0_emission = np.ones(self.num_elem, dtype=np.float32) * np.float32(t0_emission)
        elif type(t0_emission) is list:
            self._t0_emission = t0_emission
            if len(self._t0_emission) < self.num_elem:
                self._t0_emission += [0.0] * (self.num_elem - len(self._t0_emission))
            elif len(self._t0_emission) > self.num_elem:
                self._t0_emission = self._t0_emission[:self.num_elem]
            self._t0_emission = np.array(self._t0_emission, dtype=np.float32)
        elif type(t0_emission) is np.ndarray:
            self._t0_emission = t0_emission
        else:
            raise ValueError("t0_emission must be either a float [numpy.float32] or a list of floats.")

        for idx_e, e in enumerate(self.elem_list):
            e.t0 = self._t0_emission[idx_e]


class SimulationProbePoint(SimulationProbe):
    """
    Classe contendo as configurações de um transdutor do tipo ponto infinitesimal.
    É uma classe derivada de ``SimulationProbe``, específica para os transdutores do tipo
    "Point".

    Parameters
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa a posição espacial do transdutor. Por padrão se assume os
            valores [0.0, 0.0, 0.0].

        freq : int, float
            Frequência central, em MHz. Por padrão, é 5 MHz.

        bw : int, float
            Banda passante, em percentual da frequência central. Por padrão,
            é 0.5 (50%).

        pulse_type : str
            Tipo do pulso de excitação. Os tipos possíveis são: ``gaussian``,
            ``cossquare``, ``hanning`` e ``hamming``. Por padrão, é
            ``gaussian``.

    Attributes
    ----------

    """

    def __init__(self, coord_center=np.zeros((1, 3)), dec=(2, 2, 2),
                 freq=5., bw=0.5, gain=1.0, pulse_type="gaussian", id="",
                 emitter="True", receiver="False", t0_emission=None, t0_reception=None):
        # Chama o construtor da classe base.
        super().__init__(coord_center, dec)

        # Identificacao do transdutor
        self._id = id

        # Numero de elementos, para manter compatibilidade com outros tipos de transdutores
        self.num_elem = 1

        # Le a configuracao se o transdutor e emissor
        if type(emitter) is str:
            self.emitters = [ eval(emitter.lower().capitalize()) ]
        else:
            raise ValueError("emitter must be a string")

        # Le a configuracao se o transdutor e receptor
        if type(receiver) is str:
            self.receivers = [ eval(receiver.lower().capitalize()) ]
        else:
            raise ValueError("receiver must be a string")

        # Tempo de atraso para emissao dos elementos. Se for um valor escalar, e assumido para todos os elementos.
        # Se for um array, deve ter um valor para cada elemento.
        # Se for um nome de um arquivo 'law',
        if t0_emission is None:
            self._t0_emission = np.float32(0.0)
        else:
            self._t0_emission = np.float32(t0_emission)

        # Tempo de atraso para recepcao dos elementos. Se for um valor escalar, e assumido para todos os elementos.
        # Se for um array, deve ter um valor para cada elemento.
        if t0_reception is None:
            self._t0_reception = np.float32(0.0)
        else:
            self._t0_reception = np.float32(t0_reception)

        # Parametros eletricos gerais do transdutor
        self._freq = freq
        self._bw = bw
        self._gain = gain
        self._pulse_type = pulse_type

    def get_freq(self, mode='common'):
        """
        Função que retorna a frequência do transdutor.

        param mode: str
            Este parâmetro define o modo de obtenção da frequência. Este parâmetro não faz sentido em um transdutor
            pontual, mas existe para ser compatível com os demais transdutores.
            "common" significa que será utilizado o parâmetro geral, utilizado por todos os elementos ativos.
            Este é o padrão.

        :return: numpy.float32
            Retorna o valor da frequência do transdutor.
        """
        return self._freq

    def get_coords(self):
        """
        Função que retorna as coordenadas reais do centro elementos ativos do transdutor,
        no formato vetorizado.

        Returns
        -------
            : array
                Array com o ponto central ativo do transdutor.
                Cada elemento desse array é a coordenada cartesiana do centro do elemento ativo do transdutor.

        """

        return self.coord_center
        
    
    def get_points_roi(self, sim_roi=SimulationROI(), simul_type="2D", dir="e"):
        """
        Função que retorna a coordenada do ponto ativo do transdutor no grid de simulação,
        no formato vetorizado.

        Returns
        -------
            : list
                Lista com 1 ponto ativo (fonte) do transdutor como índices de pontos na ROI.
                É a coordenada cartesiana (como índice) de um ponto na ROI.

        """
        if type(dir) is str:
            if (dir.lower() == "e" and not self.emitters[0]) or (dir.lower() == "r" and not self.receivers[0]):
                return list(), list()
        else:
            raise ValueError("'dir' must be a string")

        # Calcula a coordenada do primeiro ponto
        x_coord = np.float32(self.coord_center[0])
        y_coord = np.float32(0.0 if simul_type.lower() == "2d" else self.coord_center[1])
        z_coord = np.float32(self.coord_center[2])

        # Pega os indices na ROI da coordenada do primeiro ponto
        point_coord = np.array([x_coord, y_coord, z_coord], np.float32)
        point_0 = sim_roi.get_nearest_grid_idx(point_coord)

        # Monta listas de pontos e indices
        list_out = [ [point_0[0], point_0[1], point_0[2]] ]
        idx_src = [ 0 ]

        return list_out, idx_src

    def get_source_term(self, samples=1000, dt=1.0, out='r', ord_der=1):
        """
        Função que retorna o sinais do termo de fonte do transdutor. Além de retornar um
        *array* com o sinal do termo de fonte de cada elemento ativo do transdutor, esta função
        também retorna uma lista com o índice do termo de fonte para cada ponto da ROI que é
        um ponto emissor.
        :param out:
        :param samples: int
            Número de amostras de tempo na simulação.
        :param dt: float
            Valor do passo de tempo na simulação.

        :return: :numpy.array
        Array contém dimensões as N amostras de tempo.
        """
        t = np.arange(samples, dtype=flt32) * dt
        ss = np.zeros(samples, dtype=flt32)
        if self.emitters[0]:
            if ord_der == 0:
                # Pulso gaussiano
                gp, _, egp = gaussian_pulse((t - self._t0_emission), fc=self._freq, bw=self._bw, retquad=True, retenv=True)
            elif ord_der == 2:
                # Segunda derivada do pulso gaussiano (Ricker)
                gp, _, egp = gaussian_second_dev_pulse((t - self._t0_emission), fc=self._freq, bw=self._bw, retquad=True, retenv=True)
            else:
                # Primeira derivada do pulso gaussiano
                gp, _, egp = gaussian_first_dev_pulse((t - self._t0_emission), fc=self._freq, bw=self._bw, retquad=True, retenv=True)
                
            eps = np.finfo(flt32).eps
            if out == 'e':
                egp[np.abs(egp) < eps] = 0.0
                ss = flt32(egp)
            else:
                gp[np.abs(gp) < eps] = 0.0
                ss = flt32(gp)

        return flt32(self._gain * ss)

    def get_idx_rec(self, sim_roi=SimulationROI(), simul_type="2D"):
        """
        Função que retorna um array com o índice do receptor para cada ponto da ROI que é um ponto receptor.
        :param simul_type:
        :param sim_roi:

        :return: list
        Lista com o índice do elemento receptor de cada ponto receptor na ROI.
        """
        return [ 0 ] if self.receivers[0] else list()

    def get_delay_rx(self):
        """
        Função que retorna o valor do atraso na recepção.

        :return: float
        Tempo de atraso de recepção, em microssegundos.
        """
        return [self._t0_reception] if self.receivers[0] else list()

    def set_t0(self, t0_emission=None):
        """
        Função que modifica o valor do atraso na emissão.

        :return: None
        """
        if t0_emission is None:
            self._t0_emission = np.float32(0.0)
        elif type(t0_emission) is np.float32 or type(t0_emission) is float:
            self._t0_emission = t0_emission
        else:
            raise ValueError("t0_emission must be either a float [numpy.float32].")

    def get_receiver_points_count(self):
        """
        Função que retorna a quantidade de pontos receptores.

        :return: float
        Quantidade de pontos receptores.
        """
        return 1 if self.receivers[0] else 0

class SimulationProbeMono(SimulationProbe):
    """
    Classe contendo as configurações de um transdutor do tipo mono-elemento.
    É uma classe derivada de ``SimulationProbe``, específica para transdutores do tipo
    "MonoCirc".

    Parameters
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa à posição espacial do centro do transdutor.
            Por padrão, [0.0, 0.0, 0.0].

        dec : tuple of int
            Número de casas decimais para arredondamento nos eixos (w, d, h).
            Por padrão, (2, 2, 2).

        radius : int, float
            Raio do elemento circular, em mm. Por padrão, é 0.25 mm.

        freq : int, float
            Frequência central, em MHz. Por padrão, é 5 MHz.

        bw : int, float
            Banda passante, em percentual da frequência central. Por padrão,
            é 0.5 (50%).

        gain : int, float
            Ganho do elemento transdutor. Por padrão, é 1.0.

        pulse_type : str
            Tipo do pulso de excitação. O único tipo possível é: ``gaussian``.
            Por padrão, é ``gaussian``.

        id : str
            Identificador do transdutor. Por padrão, é "".

        emitter : str
            String booleana indicando se o transdutor é emissor ("True" ou "False").
            Por padrão, é "True".

        receiver : str
            String booleana indicando se o transdutor é receptor ("True" ou "False").
            Por padrão, é "False".

        t0_emission : float, optional
            Atraso do sinal de emissão, em microssegundos. Por padrão, é 0.0.

        t0_reception : float, optional
            Atraso do sinal de recepção, em microssegundos. Por padrão, é 0.0.

    Attributes
    ----------
        num_elem : int
            Número de elementos. Sempre igual a 1 para este transdutor.

        elem : :class:`ElementCirc`
            Objeto do tipo ``ElementCirc`` contendo as características físicas e
            elétricas do elemento ativo do transdutor.

        emitters : list of bool
            Lista com flag indicando se o elemento é emissor.

        receivers : list of bool
            Lista com flag indicando se o elemento é receptor.
    """

    def __init__(self, coord_center=np.zeros((1, 3)), dec=(2, 2, 2),
                 radius=3.175,
                 freq=5., bw=0.5, gain=1.0, pulse_type="gaussian", id="",
                 emitter="True", receiver="False",
                 t0_emission=None, t0_reception=None):

        # Chama o construtor da classe base.
        super().__init__(coord_center, dec)

        # Identificação do transdutor.
        self._id = id

        # Número de elementos (sempre 1 para transdutor mono).
        self.num_elem = 1

        # Lê a configuração se o transdutor é emissor.
        if type(emitter) is str:
            self.emitters = [eval(emitter.lower().capitalize())]
        else:
            raise ValueError("emitter must be a string")

        # Lê a configuração se o transdutor é receptor.
        if type(receiver) is str:
            self.receivers = [eval(receiver.lower().capitalize())]
        else:
            raise ValueError("receiver must be a string")

        # Tempo de atraso para emissão.
        if t0_emission is None:
            self._t0_emission = np.float32(0.0)
        else:
            self._t0_emission = np.float32(t0_emission)

        # Tempo de atraso para recepção.
        if t0_reception is None:
            self._t0_reception = np.float32(0.0)
        else:
            self._t0_reception = np.float32(t0_reception)

        # Instancia o elemento circular único.
        # coord_center do elemento é zerado pois o offset do probe é aplicado em get_points_roi.
        self.elem_list = [ElementCirc(
            radius=radius,
            coord_center=np.zeros(3, dtype=np.float32),
            freq=freq,
            bw=bw,
            gain=gain,
            t0=self._t0_emission,
            tx_en=self.emitters[0],
            rx_en=self.receivers[0],
            pulse_type=pulse_type
            )
        ]

        # Parâmetros gerais do transdutor.
        self._radius = np.float32(radius)
        self._freq = freq
        self._bw = bw
        self._gain = gain
        self._pulse_type = pulse_type

    def get_freq(self, mode='common'):
        """
        Retorna a frequência do transdutor.

        Parameters
        ----------
        mode : str

        Returns
        -------
        numpy.float32
            Valor da frequência do transdutor.
        """
        return np.float32(self._freq)

    def get_coords(self):
        """
        Retorna a coordenada real do centro do elemento ativo do transdutor.

        Returns
        -------
        np.ndarray
            Coordenada cartesiana do centro do elemento ativo.
        """
        return self.coord_center

    def get_points_roi(self, sim_roi=SimulationROI(), simul_type="2D", dir="e"):
        """
        Função que retorna a coordenada do ponto ativo do transdutor no grid de simulação,
        no formato vetorizado.

        """
        arr_out = list()
        idx_src = list()
        for idx_st, e in enumerate(self.elem_list):
            try:
                arr_elem = e.get_points_roi(sim_roi=sim_roi, probe_center=self.coord_center,
                                            simul_type=simul_type, dir=dir)
                arr_out += arr_elem
                if len(arr_elem):
                    idx_src += [idx_st for _ in range(len(arr_elem))]
            except IndexError:
                pass

        return arr_out, idx_src

    def get_source_term(self, samples=1000, dt=1.0, out='r'):
        """
        Retorna o sinal do termo de fonte do transdutor.

        """
        dec = int(abs(np.log10(dt))) + 2
        t = np.round(np.arange(samples, dtype=np.float32) * dt, decimals=dec)
        source_term = np.zeros((samples, self.num_elem), dtype=np.float32)
        
        for idx_st, e in enumerate(self.elem_list):
            if e.tx_en:
                source_term[:, idx_st] = e.get_element_exc_fn(t, out)

        return source_term

    def get_idx_rec(self, sim_roi=SimulationROI(), simul_type="2D"):
        """
        Função que retorna um array com o índice do receptor para cada ponto da ROI que é um ponto receptor.

        """
        idx_rec = list()
        for idx_st, e in enumerate(self.elem_list):
            try:
                arr_elem = e.get_points_roi(sim_roi=sim_roi, probe_center=self.coord_center,
                                            simul_type=simul_type, dir='r')
                if len(arr_elem):
                    idx_rec += [idx_st for _ in range(len(arr_elem))]
            except IndexError:
                pass

        return idx_rec

    def get_delay_rx(self):
        """
        Retorna o valor do atraso na recepção.

        """
        t0_recp = list()
        t0_recp.append(self._t0_reception)

        return t0_recp

    def set_t0(self, t0_emission=None):
        """
        Modifica o valor do atraso na emissão, atualizando também o elemento interno.

        """
        if t0_emission is None:
            self._t0_emission = np.zeros(self.num_elem, dtype=np.float32)
        elif type(t0_emission) is np.float32 or type(t0_emission) is float:
            self._t0_emission = np.ones(self.num_elem, dtype=np.float32) * np.float32(t0_emission)
        elif type(t0_emission) is list:
            self._t0_emission = t0_emission
            if len(self._t0_emission) < self.num_elem:
                self._t0_emission += [0.0] * (self.num_elem - len(self._t0_emission))
            elif len(self._t0_emission) > self.num_elem:
                self._t0_emission = self._t0_emission[:self.num_elem]
            self._t0_emission = np.array(self._t0_emission, dtype=np.float32)
        elif type(t0_emission) is np.ndarray:
            self._t0_emission = t0_emission
        else:
            raise ValueError("t0_emission must be either a float [numpy.float32] or a list of floats.")

        for idx_e, e in enumerate(self.elem_list):
            e.t0 = self._t0_emission[idx_e]

    def get_receiver_points_count(self):
        """
        Retorna a quantidade de pontos receptores do transdutor.

        """
        return len(self.get_idx_rec(sim_roi=sim_roi, simul_type=simul_type))