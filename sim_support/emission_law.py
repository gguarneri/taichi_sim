import numpy as np

class EmissionLaw:
    def __init__(self):
        # Indices de cada informacao no arquivo '.law'
        self._idx_numR = 0
        self._idx_numS = 1
        self._idx_numT = 2
        self._idx_numL = 3
        self._idx_numV = 4
        self._idx_retE = 5
        self._idx_ampE = 6
        self._idx_retR = 7
        self._idx_ampR = 8
    
    def __extract_line_info(self, line):
        return line[:-1].split('\t')


    def __lines2numpyarray(self, lines):
        num_lines = len(lines)
        return np.array([self.__extract_line_info(lines[i]) for i in range(num_lines)], dtype=np.float32)
    
    def write_law(self, root, emitter_law, receiver_law=None, emitter_amp=None, reciever_amp=None, elem_range=None):
        header = [
            "# LOIS DE RETARD \n",
            "Version 1.0 \n",
            "numR\t"
            "numS\t"
            "numT\t"
            "numL\t"
            "numV\t"
            "retE\t"
            "ampE\t"
            "retR\t"
            "ampR\n"
        ]

        if elem_range is None:
            elem_range = [0, emitter_law.shape[1] - 1]

        if receiver_law is None:
            receiver_law = emitter_law
            
        with open(root + ".law", "w") as file:
            file.writelines(header)
            for shot in range(0, emitter_law.shape[0]):
                for elem_idx in range(elem_range[0], elem_range[-1] + 1):
                    numR = 0  #
                    numS = 0  #
                    numT = shot  # Shot
                    numL = 0  #
                    numV = elem_idx + 1  # Indice do Emissor
                    retE = emitter_law[shot, elem_idx]  # Lei focal na Emissao
                    if emitter_amp is not None:
                        ampE = emitter_amp[shot, elem_idx]
                    else:
                        ampE = 1  # Ganho na Emissao
                    retR = receiver_law[shot, elem_idx]  # Lei focal na Recepcao
                    if reciever_amp is not None:
                        ampR = reciever_amp[shot, elem_idx]
                    else:
                        ampR = 1  # Ganho na Recepcao
                    datum = [numR, numS, numT, numL, numV, retE, ampE, retR, ampR]
                    data_line = [f"{datum[i]}" + "\t" for i in range(0, len(datum) - 1)]
                    data_line.append(f"{datum[-1]}\n")
                    file.writelines(data_line)

    def read_law(self, file_root):
        with open(file_root) as f:
            # Le o arquivo .law
            lines_with_header = f.readlines()
            
            # Extrai os parametros do cabeçalho
            header = self.__extract_line_info(lines_with_header[2])
            num_data = len(header)
            lines = lines_with_header[3:]
            lines_data = self.__lines2numpyarray(lines)

            num_elements = int(np.max(lines_data[:, self._idx_numV]))  # O numero de elementos sendo pulsados
            max_numS = int(np.max(lines_data[:, self._idx_numS]))
            if max_numS != 0:
                num_shots = int(np.max(lines_data[:, self._idx_numS]) + 1)  # E um FMC
            elif max_numS == 0:
                num_shots = int(np.max(lines_data[:, self._idx_numT]) + 1)  # O numero de disparos/shots
            else:
                raise ValueError("Valor nao suportado para o parametro numS no arquivo '.law'")

            delay_law = np.zeros(shape=(num_shots, num_elements), dtype='float')
            amplitude_law = np.zeros(shape=(num_shots, num_elements), dtype='float')

            for shot_idx in range(num_shots):
                i_beg = shot_idx * num_elements
                i_end = (shot_idx + 1) * num_elements

                # Le todos os parametros
                current_shot_numR = lines_data[i_beg:i_end, self._idx_numR]  #
                current_shot_numS = lines_data[i_beg:i_end, self._idx_numS]  #
                current_shot_numT = lines_data[i_beg:i_end, self._idx_numT]  # Shot
                current_shot_numL = lines_data[i_beg:i_end, self._idx_numL]  #
                current_shot_numV = lines_data[i_beg:i_end, self._idx_numV]  # Indice do Emissor
                current_shot_retE = lines_data[i_beg:i_end, self._idx_retE]  # Lei focal na Emissao
                current_shot_ampE = lines_data[i_beg:i_end, self._idx_ampE]  # Ganho na Emissao
                current_shot_retR = lines_data[i_beg:i_end, self._idx_retR]  # Lei focal na Recepcao
                current_shot_ampR = lines_data[i_beg:i_end, self._idx_ampR]  # Ganho na Recepcao

                # Assumindo que o delay da transmissao e igual ao da recepcao
                delay_law[shot_idx, :] = current_shot_retE  # Assumindo que current_shot_retE == current_shot_retT

                # Assumindo que a amplitude da transmissao e igual ao da recepcao
                amplitude_law[shot_idx, :] = current_shot_ampE  # Assumindo que current_shot_ampE == current_shot_ampT

        return delay_law, amplitude_law   