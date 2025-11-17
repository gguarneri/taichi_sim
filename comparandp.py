import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k

def criador_mapas():
    cs_map = np.zeros((600,800), dtype=np.float32)
    cp_map = np.zeros((600,800), dtype=np.float32)
    rho_map = np.zeros((600,800), dtype=np.float32)

    #aluminio
    cs_map[:,:] = 3.1965
    cp_map[:,:] = 6.393
    rho_map[:,:] = 2700.0
    #agua
    # cs_map[:,4001:] = 0.0
    # cp_map[:,4001:] = 1.483
    # rho_map[:,4001:] = 1000.0
    #ar
    # cs_map[0:10,:] = 0.0
    # cs_map[610:,:] = 0.0
    # cp_map[0:10,:] = 0.346
    # cp_map[610:,:] = 0.346
    # rho_map[0:10,:] = 1.201
    # rho_map[610:,:] = 1.201
    #
    # #buraco de ar
    # cs_map[205:215,398:402] = 0.0
    # cs_map[208:212,395:405] = 0.0
    # cs_map[207:213,396:404] = 0.0
    # cs_map[206:214, 397:403] = 0.0
    #
    # cp_map[205:215,398:402] = 0.346
    # cp_map[208:212,395:405] = 0.346
    # cp_map[207:213,396:404] = 0.346
    # cp_map[206:214, 397:403] = 0.346
    #
    # rho_map[205:215,398:402] = 1.201
    # rho_map[208:212,395:405] = 1.201
    # rho_map[207:213,396:404] = 1.201
    # rho_map[206:214,  397:403] = 1.201

    plt.figure(1)
    plt.imshow(cs_map, cmap='gray')
    plt.show()
    np.save('./ensaios/ensaio_tat/tat_maps/bloq_nh_600x800_cs.npy',cs_map)
    np.save('./ensaios/ensaio_tat/tat_maps/bloq_nh_600x800_cp.npy', cp_map)
    np.save('./ensaios/ensaio_tat/tat_maps/bloq_nh_600x800_rho.npy', rho_map)
    teste = cs_map


def comparacao():

        amostra1 = np.load(
            './ensaios/bloquinho/results/result_WebGPU-viscoelastic_20251114-085909_826x626_5000_iter_0_law_0_bscan_stress.npy')
        amostra2 = np.load(
            './ensaios/bloquinho/results/result_WebGPU-viscoelastic_20251114-090018_826x626_5000_iter_0_law_0_bscan_stress.npy')
        erro = 0
        for i in range(5000):
            for j in range(64):
                if amostra1[i,j] == amostra2[i,j]:
                    erro = erro + 1
                    print(erro)
                    print(i)
                    print(j)

def comp_att():
    result_viscoelast = np.load('./ensaios/ensaio_tat/results/nh_att_std.npy')
    result_viscoelast2 = np.load('ensaios/ensaio_tat/results/att_48.npy')
    result_elast = np.load('./ensaios/ensaio_tat/results/nh_elast.npy')
    data = file_m2k.read(r"C:\Users\Henrique\PycharmProjects\webgpu_sim\H_facin_teste2.m2k", 5, 0.5, 'Gaussian')
    result_real = data.ascan_data[:, 0, :, 0]

    ascan_viscoelast = result_viscoelast[450:,4] / (np.abs(result_viscoelast[450:,4]).max())
    ascan_elast = result_elast[450:,4] / (np.abs(result_elast[450:,4]).max())
    ascan_real = result_real[:6750,4] / (np.abs(result_real[:6750,4]).max())
    ascan_viscoelast2 = result_viscoelast2[450:,4] / (np.abs(result_viscoelast2[450:,4]).max())

    t_real = np.linspace(0, 54, len(ascan_real))
    t_visco2 = np.linspace(0, 54, len(ascan_viscoelast2))

    plt.figure(0)
    plt.plot(t_real, ascan_real, label='Real', alpha=0.8, color='green')
    plt.plot(t_visco2, ascan_viscoelast2, label='Visco', alpha=0.6, color='red', linestyle='--' )
    plt.grid(True, alpha=0.3)

    plt.figure(1)
    plt.title('viscoelastic stress')
    plt.plot(ascan_viscoelast, color='red')
    plt.figure(2)
    plt.title('elastic stress')
    plt.plot(ascan_elast, color='blue')
    plt.figure(3)
    plt.title('real nde')
    plt.plot(ascan_real, color='green')
    plt.figure(4)
    plt.title('viscoelastic_v2')
    plt.plot(ascan_viscoelast2, color='red')



    plt.show()


comp_att()
print("Olá mundo")


# #aluminio
# cs_map[51:651,51:851] = 3.1965
# cp_map[51:651,51:851] = 6.393
# rho_map[51:651,51:851] = 2700.0
# #agua
# # cs_map[:,4001:] = 0.0
# # cp_map[:,4001:] = 1.483
# # rho_map[:,4001:] = 1000.0
# #ar
# cs_map[0:51,0:51] = 0.0
# cs_map[651:,851:] = 0.0
# cp_map[0:51,0:51] = 0.346
# cp_map[651:,851:] = 0.346
# rho_map[0:51,0:51] = 1.201
# rho_map[651:,851:] = 1.201
#
# #buraco de ar
# cs_map[245:255,448:452] = 0.0
# cs_map[248:252,445:455] = 0.0
# cs_map[247:253,446:454] = 0.0
# cs_map[246:254, 447:453] = 0.0
#
# cp_map[245:255,448:452] = 0.346
# cp_map[248:252,445:455] = 0.346
# cp_map[247:253,446:454] = 0.346
# cp_map[246:254, 447:453] = 0.346
#
# rho_map[245:255,448:452] = 1.201
# rho_map[248:252,445:455] = 1.201
# rho_map[247:253,446:454] = 1.201
# rho_map[246:254, 447:453] = 1.201
#
# plt.imshow(cs_map, cmap='gray')
# plt.show()
# np.save('./ensaios/ensaio_tat/tat_maps/bloq_tat_700x900_cs.npy',cs_map)
# np.save('./ensaios/ensaio_tat/tat_maps/bloq_tat_700x900_cp.npy', cp_map)
# np.save('./ensaios/ensaio_tat/tat_maps/bloq_tat_700x900_rho.npy', rho_map)