# =======================
# Importacao de pacotes de uso geral
# =======================
import argparse
from time import time
import numpy as np
from sim_support import *
from sim_support.simulator import Simulator

# ======================
# Importacao de pacotes especificos para a implementacao do simulador
# ======================
import wgpu


# -----------------------------------------------------------------------------
# Aqui deve ser implementado o simulador como uma classe herdada de Simulator
# -----------------------------------------------------------------------------
class SimulatorWebGPU(Simulator):
    def __init__(self, file_config):
        # Chama do construtor padrao, que le o arquivo de configuracao
        super().__init__(file_config)
        
        # Define o nome do simulador
        self._name = "WebGPU"
        
        # -----------------------
        # Inicializacao do WebGPU
        # -----------------------
        self._device = None
        self._gpu_type = self._configs["simul_configs"]["gpu_type"] if "gpu_type" in self._configs["simul_configs"] else "high-perf"
        if self._gpu_type == "high-perf":
            self._device = wgpu.utils.get_default_device()
        else:
            self._device = wgpu.gpu.request_adapter(power_preference="low-power").request_device()

        # Escolha dos valores de wsx e wsy
        self._wsx = np.gcd(self._roi.get_nx(), 16)
        self._wsy = np.gcd(self._roi.get_nz(), 16)
        
    def implementation(self):
        super().implementation()
        
        # --------------------------------------------
        # Aqui comeca o codigo especifico do simulador
        # --------------------------------------------
        # Cria o shader para calculo contido no arquivo ``shader_webgpu.wgsl''
        with open('shader_webgpu.wgsl') as shader_file:
            cshader_string = shader_file.read()
            cshader_string = cshader_string.replace('_WSX_', f'{self._wsx}')
            cshader_string = cshader_string.replace('_WSY_', f'{self._wsy}')
            cshader_string = cshader_string.replace('_IDX_REC_OFFSET_', f'{self._idx_rec_offset}')
            cshader = self._device.create_shader_module(code=cshader_string)
            
        # Arrays com parametros inteiros (i32) e ponto flutuante (f32) para rodar o simulador
        ord = self._coefs.shape[0]
        params_i32 = np.array([self._nx, self._ny, self._n_steps, self._n_src, self._n_rec, self._n_pto_rec, ord, 0], dtype=int32)
        params_f32 = np.array([self._roi.get_dx(), self._roi.get_dz(), self._dt], dtype=flt32)

        # Definicao dos buffers que terao informacoes compartilhadas entre CPU e GPU
        # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
        read_only_mask = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
        read_write_mask = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        
        # ------- Buffers para o binding de parametros -------------
        # Buffer de parametros com valores inteiros
        b_param_int32 = self._device.create_buffer_with_data(data=params_i32, usage=read_write_mask)
        
        # Buffer de parametros com valores em ponto flutuante
        b_param_flt32 = self._device.create_buffer_with_data(data=params_f32, usage=read_only_mask)

        # Forcas da fonte
        b_force = self._device.create_buffer_with_data(data=self._source_term, usage=read_only_mask)

        # Indices das fontes na ROI
        b_idx_src = self._device.create_buffer_with_data(data=self._pos_sources, usage=read_only_mask)

        # Coeficientes de absorcao
        b_a_x = self._device.create_buffer_with_data(data=self._a_x.flatten(), usage=read_only_mask)
        b_b_x = self._device.create_buffer_with_data(data=self._b_x.flatten(), usage=read_only_mask)
        b_k_x = self._device.create_buffer_with_data(data=self._k_x.flatten(), usage=read_only_mask)
        b_a_x_h = self._device.create_buffer_with_data(data=self._a_x_half.flatten(), usage=read_only_mask)
        b_b_x_h = self._device.create_buffer_with_data(data=self._b_x_half.flatten(), usage=read_only_mask)
        b_k_x_h = self._device.create_buffer_with_data(data=self._k_x_half.flatten(), usage=read_only_mask)
        b_a_y = self._device.create_buffer_with_data(data=self._a_y.flatten(), usage=read_only_mask)
        b_b_y = self._device.create_buffer_with_data(data=self._b_y.flatten(), usage=read_only_mask)
        b_k_y = self._device.create_buffer_with_data(data=self._k_y.flatten(), usage=read_only_mask)
        b_a_y_h = self._device.create_buffer_with_data(data=self._a_y_half.flatten(), usage=read_only_mask)
        b_b_y_h = self._device.create_buffer_with_data(data=self._b_y_half.flatten(), usage=read_only_mask)
        b_k_y_h = self._device.create_buffer_with_data(data=self._k_y_half.flatten(), usage=read_only_mask)

        # Buffers com os indices para o calculo com o staggered grid
        idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(ord)], dtype=int32)
        b_idx_fd = self._device.create_buffer_with_data(data=idx_fd, usage=read_only_mask)

        # Buffer com os mapas de velocidade e densidade da ROI
        b_rho_map = self._device.create_buffer_with_data(data=self._rho_grid_vx, usage=read_only_mask)
        b_cp_map = self._device.create_buffer_with_data(data=self._cp_grid_vx, usage=read_only_mask)

        # Buffer com os coeficientes para ao calculo das derivadas
        b_fd_coeffs = self._device.create_buffer_with_data(data=self._coefs, usage=read_only_mask)

        # Buffers com os arrays de simulacao
        # ----------------- Velocidades ---------------------
        # Arrays para as variaveis de memoria do calculo
        b_vx = self._device.create_buffer_with_data(data=np.zeros((self._nx, self._ny), dtype=flt32), usage=read_write_mask)
        b_vy = self._device.create_buffer_with_data(data=np.zeros((self._nx, self._ny), dtype=flt32), usage=read_write_mask)
        b_p_2 = self._device.create_buffer_with_data(data=flt32(0.0), usage=read_write_mask)

        # ----------------- Pressao ---------------------
        b_pressure = self._device.create_buffer_with_data(data=np.zeros((self._nx, self._ny), dtype=flt32), usage=read_write_mask)
        
        # Arrays de memoria do simulador
        b_memory_dvx_dx = self._device.create_buffer_with_data(data=np.zeros((self._nx, self._ny), dtype=flt32), usage=read_only_mask)
        b_memory_dvy_dy = self._device.create_buffer_with_data(data=np.zeros((self._nx, self._ny), dtype=flt32), usage=read_only_mask)
        b_memory_dpressure_dx = self._device.create_buffer_with_data(data=np.zeros((self._nx, self._ny), dtype=flt32), usage=read_only_mask)
        b_memory_dpressure_dy = self._device.create_buffer_with_data(data=np.zeros((self._nx, self._ny), dtype=flt32), usage=read_only_mask)

        # Sinais dos sensores
        b_sens_x = self._device.create_buffer_with_data(data=np.zeros((self._n_steps, self._n_rec), dtype=flt32), usage=read_write_mask)
        b_sens_y = self._device.create_buffer_with_data(data=np.zeros((self._n_steps, self._n_rec), dtype=flt32), usage=read_write_mask)
        b_sens_pressure = self._device.create_buffer_with_data(data=np.zeros((self._n_steps, self._n_rec), dtype=flt32), usage=read_write_mask)

        # Tempo de espera para recepcao nos sensores
        b_delay_rec = self._device.create_buffer_with_data(data=self._delay_recv, usage=read_only_mask)

        # Informacoes dos pontos receptores
        b_info_rec_pt = self._device.create_buffer_with_data(data=self._info_rec_pt, usage=read_only_mask)
        b_offset_sensors = self._device.create_buffer_with_data(data=self._offset_sensors, usage=read_only_mask)

        # Esquema de amarracao dos parametros (binding layouts [bl])
        # Parametros
        bl_params = [
            {"binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage}
            }
        ]
        bl_params += [
            {"binding": ii,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage}
            } for ii in range(1, 20)
        ]

        # Arrays da simulacao
        bl_sim_arrays = [
            {"binding": ii,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage}
            } for ii in range(0, 8)
        ]

        # Sensores
        bl_sensors = [
            {"binding": ii,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage}
            } for ii in range(0, 3)
        ]
        bl_sensors += [
            {"binding": ii,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage}
            } for ii in range(3, 6)
        ]

        # Configuracao das amarracoes (bindings)
        b_params = [
            {
                "binding": 0,
                "resource": {"buffer": b_param_int32, "offset": 0, "size": b_param_int32.size},
            },
            {
                "binding": 1,
                "resource": {"buffer": b_param_flt32, "offset": 0, "size": b_param_flt32.size},
            },
            {
                "binding": 2,
                "resource": {"buffer": b_force, "offset": 0, "size": b_force.size},
            },
            {
                "binding": 3,
                "resource": {"buffer": b_idx_src, "offset": 0, "size": b_idx_src.size},
            },
            {
                "binding": 4,
                "resource": {"buffer": b_a_x, "offset": 0, "size": b_a_x.size},
            },
            {
                "binding": 5,
                "resource": {"buffer": b_b_x, "offset": 0, "size": b_b_x.size},
            },
            {
                "binding": 6,
                "resource": {"buffer": b_k_x, "offset": 0, "size": b_k_x.size},
            },
            {
                "binding": 7,
                "resource": {"buffer": b_a_x_h, "offset": 0, "size": b_a_x_h.size},
            },
            {
                "binding": 8,
                "resource": {"buffer": b_b_x_h, "offset": 0, "size": b_b_x_h.size},
            },
            {
                "binding": 9,
                "resource": {"buffer": b_k_x_h, "offset": 0, "size": b_k_x_h.size},
            },
            {
                "binding": 10,
                "resource": {"buffer": b_a_y, "offset": 0, "size": b_a_y.size},
            },
            {
                "binding": 11,
                "resource": {"buffer": b_b_y, "offset": 0, "size": b_b_y.size},
            },
            {
                "binding": 12,
                "resource": {"buffer": b_k_y, "offset": 0, "size": b_k_y.size},
            },
            {
                "binding": 13,
                "resource": {"buffer": b_a_y_h, "offset": 0, "size": b_a_y_h.size},
            },
            {
                "binding": 14,
                "resource": {"buffer": b_b_y_h, "offset": 0, "size": b_b_y_h.size},
            },
            {
                "binding": 15,
                "resource": {"buffer": b_k_y_h, "offset": 0, "size": b_k_y_h.size},
            },
            {
                "binding": 16,
                "resource": {"buffer": b_idx_fd, "offset": 0, "size": b_idx_fd.size},
            },
            {
                "binding": 17,
                "resource": {"buffer": b_fd_coeffs, "offset": 0, "size": b_fd_coeffs.size},
            },
            {
                "binding": 18,
                "resource": {"buffer": b_rho_map, "offset": 0, "size": b_rho_map.size},
            },
            {
                "binding": 19,
                "resource": {"buffer": b_cp_map, "offset": 0, "size": b_cp_map.size},
            },
        ]
        b_sim_arrays = [
            {
                "binding": 0,
                "resource": {"buffer": b_vx, "offset": 0, "size": b_vx.size},
            },
            {
                "binding": 1,
                "resource": {"buffer": b_vy, "offset": 0, "size": b_vy.size},
            },
            {
                "binding": 2,
                "resource": {"buffer": b_p_2, "offset": 0, "size": b_p_2.size},
            },
            {
                "binding": 3,
                "resource": {"buffer": b_pressure, "offset": 0, "size": b_pressure.size},
            },
            {
                "binding": 4,
                "resource": {"buffer": b_memory_dvx_dx, "offset": 0, "size": b_memory_dvx_dx.size},
            },
            {
                "binding": 5,
                "resource": {"buffer": b_memory_dvy_dy, "offset": 0, "size": b_memory_dvy_dy.size},
            },
            {
                "binding": 6,
                "resource": {"buffer": b_memory_dpressure_dx, "offset": 0, "size": b_memory_dpressure_dx.size},
            },
            {
                "binding": 7,
                "resource": {"buffer": b_memory_dpressure_dy, "offset": 0, "size": b_memory_dpressure_dy.size},
            },
        ]
        b_sensors = [
            {
                "binding": 0,
                "resource": {"buffer": b_sens_x, "offset": 0, "size": b_sens_x.size},
            },
            {
                "binding": 1,
                "resource": {"buffer": b_sens_y, "offset": 0, "size": b_sens_y.size},
            },
            {
                "binding": 2,
                "resource": {"buffer": b_sens_pressure, "offset": 0, "size": b_sens_pressure.size},
            },
            {
                "binding": 3,
                "resource": {"buffer": b_delay_rec, "offset": 0, "size": b_delay_rec.size},
            },
            {
                "binding": 4,
                "resource": {"buffer": b_info_rec_pt, "offset": 0, "size": b_info_rec_pt.size},
            },
            {
                "binding": 5,
                "resource": {"buffer": b_offset_sensors, "offset": 0, "size": b_offset_sensors.size},
            },
        ]

        # Coloca tudo junto
        bgl_0 = self._device.create_bind_group_layout(entries=bl_params)
        bgl_1 = self._device.create_bind_group_layout(entries=bl_sim_arrays)
        bgl_2 = self._device.create_bind_group_layout(entries=bl_sensors)
        pipeline_layout = self._device.create_pipeline_layout(bind_group_layouts=[bgl_0, bgl_1, bgl_2])
        bg_0 = self._device.create_bind_group(layout=bgl_0, entries=b_params)
        bg_1 = self._device.create_bind_group(layout=bgl_1, entries=b_sim_arrays)
        bg_2 = self._device.create_bind_group(layout=bgl_2, entries=b_sensors)

        # Cria os pipelines de execucao
        compute_teste_kernel = self._device.create_compute_pipeline(layout=pipeline_layout,
                                                                    compute={"module": cshader,
                                                                             "entry_point": "teste_kernel"})
        compute_pressure_kernel = self._device.create_compute_pipeline(layout=pipeline_layout,
                                                                       compute={"module": cshader,
                                                                                "entry_point": "pressure_kernel"})
        compute_sources_kernel = self._device.create_compute_pipeline(layout=pipeline_layout,
                                                                      compute={"module": cshader,
                                                                               "entry_point": "sources_kernel"})
        compute_velocity_kernel = self._device.create_compute_pipeline(layout=pipeline_layout,
                                                                       compute={"module": cshader,
                                                                                "entry_point": "velocity_kernel"})
        compute_finish_it_kernel = self._device.create_compute_pipeline(layout=pipeline_layout,
                                                                        compute={"module": cshader,
                                                                                 "entry_point": "finish_it_kernel"})
        compute_store_sensors_kernel = self._device.create_compute_pipeline(layout=pipeline_layout,
                                                                            compute={"module": cshader,
                                                                                     "entry_point": "store_sensors_kernel"})
        compute_incr_it_kernel = self._device.create_compute_pipeline(layout=pipeline_layout,
                                                                      compute={"module": cshader,
                                                                               "entry_point": "incr_it_kernel"})

        # Definicao dos limites para a plotagem dos campos
        v_max = 100.0
        v_min = - v_max
        ix_min = self._roi.get_ix_min()
        ix_max = self._roi.get_ix_max()
        iy_min = self._roi.get_iz_min()
        iy_max = self._roi.get_iz_max()

        # Laco de tempo para execucao da simulacao
        t_gpu = time()
        for it in range(1, self._n_steps + 1):
            # Cria o codificador de comandos
            command_encoder = self._device.create_command_encoder()

            # Inicia os passos de execucao do decodificador
            compute_pass = command_encoder.begin_compute_pass()

            # Ajusta os grupos de amarracao
            compute_pass.set_bind_group(0, bg_0, [], 0, 999999)  # last 2 elements not used
            compute_pass.set_bind_group(1, bg_1, [], 0, 999999)  # last 2 elements not used
            compute_pass.set_bind_group(2, bg_2, [], 0, 999999)  # last 2 elements not used

            # Ativa o pipeline de teste
            # compute_pass.set_pipeline(compute_teste_kernel)
            # compute_pass.dispatch_workgroups(self._nx // self._wsx, self._ny // self._wsy)

            # # Ativa o pipeline de execucao do calculo da pressao
            compute_pass.set_pipeline(compute_pressure_kernel)
            compute_pass.dispatch_workgroups(self._nx // self._wsx, self._ny // self._wsy)

            # Ativa o pipeline de adicao das fontes no campo de pressao
            compute_pass.set_pipeline(compute_sources_kernel)
            compute_pass.dispatch_workgroups(self._nx // self._wsx, self._ny // self._wsy)
            
            # Ativa o pipeline de execucao do calculo das velocidades
            compute_pass.set_pipeline(compute_velocity_kernel)
            compute_pass.dispatch_workgroups(self._nx // self._wsx, self._ny // self._wsy)

            # Ativa o pipeline de execucao dos procedimentos finais da iteracao
            compute_pass.set_pipeline(compute_finish_it_kernel)
            compute_pass.dispatch_workgroups(self._nx // self._wsx, self._ny // self._wsy)

            # Ativa o pipeline de execucao do armazenamento dos sensores
            compute_pass.set_pipeline(compute_store_sensors_kernel)
            compute_pass.dispatch_workgroups(self._idx_rec_offset)

            # Ativa o pipeline de atualizacao da amostra de tempo
            compute_pass.set_pipeline(compute_incr_it_kernel)
            compute_pass.dispatch_workgroups(1)

            # Termina o passo de execucao
            compute_pass.end()

            # Efetua a execucao dos comandos na GPU
            self._device.queue.submit([command_encoder.finish()])

            # Leitura da GPU para sincronismo
            psn2 = np.sqrt(self._device.queue.read_buffer(b_p_2, buffer_offset=0, size=b_p_2.size).cast("f")[0])
            if (it % self._it_display) == 0 or it == 5:
                if self._show_debug:
                    print(f'Time step # {it} out of {self._n_steps}')
                    print(f'Max pressure = {psn2}')

                if self._show_anim:
                    pressuregpu = np.asarray(self._device.queue.read_buffer(b_pressure, buffer_offset=0).cast("f")).reshape((self._nx, self._ny))
                    self._windows_gpu[0].imv.setImage(pressuregpu[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    self._app.processEvents()

            # Verifica a estabilidade da simulacao
            if psn2 > STABILITY_THRESHOLD:
                raise StabilityError("Simulacao tornando-se instavel", psn2)
                
        sim_time = time() - t_gpu

        # Pega os resultados da simulacao
        pressure = np.asarray(self._device.queue.read_buffer(b_pressure, buffer_offset=0).cast("f")).reshape((self._nx, self._ny))
        sens_pressure = np.array(self._device.queue.read_buffer(b_sens_pressure).cast("f")).reshape((self._n_steps, self._n_rec))

        # --------------------------------------------
        # A funcao de implementacao do simulador deve retornar
        # um dicionario com as seguintes chaves:
        #   - "pressure": campo de pressao
        #   - "sens_pressure": sinais da pressao nos sensores
        #   - "gpu_str": string de identificacao da GPU utilizada na simulacao
        #   - "sim_time": tempo da simulacao, medido com a funcao time()
        #   - opcionalmente pode ter uma mensagem exclusiva da implementacao em "msg_impl"
        # --------------------------------------------
        return {"pressure": pressure, "sens_pressure": sens_pressure,
                "gpu_str": self._device.adapter.info["device"], "sim_time": sim_time}
        

# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# Cria a instancia do simulador
sim_instance = SimulatorWebGPU(args.config)

# Executa simulacao
try:
    sim_instance.run()

except KeyError as key:
    print(f"Chave {key} nao encontrada no arquivo de configuracao.")
    
except ValueError as value:
    print(value)
