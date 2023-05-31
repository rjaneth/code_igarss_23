import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import polsar_total_loglikelihood as ptl
#import cv2
import os.path
import scipy.stats as stats
import scipy.special
from scipy.special import softmax
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

#from PIL import Image
#
#from mpl_toolkits.mplot3d import axes3d, Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator


# ORIGINAL
# def show_evidence(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
#     PIA=pauli.copy()
    
#     plt.figure(figsize=(20*img_rt, 20))
#     for k in range(NUM_RAIOS):
#         ik = np.int(evidence[k, banda])
#         ia = np.int(MXC[k, ik])
#         ja = np.int(MYC[k, ik])
#         plt.plot(ia, ja, marker='o', color="deepskyblue")
#     plt.imshow(PIA)
#     plt.grid(None)
#     # Guardar la imagen en formato PDF
#     plt.savefig('./datos/recorte_flev_hv.pdf', format='pdf', dpi=300, bbox_inches='tight')
#     #plt.savefig('./datos/flev_hh1.pdf', format='pdf', dpi=300)
#     plt.show() 
#     return PIA

#----NUEVO
def show_evidence(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
    PIA = pauli.copy()
    coordenadas = np.zeros((NUM_RAIOS, 2))  # Crear arreglo para almacenar las coordenadas (x, y)
    
    plt.figure(figsize=(20*img_rt, 20))
    for k in range(NUM_RAIOS):
        ik = np.int(evidence[k, banda])
        ia = np.int(MXC[k, ik])
        ja = np.int(MYC[k, ik])
        plt.plot(ia, ja, marker='o', color="deepskyblue")
        
        # Almacenar las coordenadas (x, y) en el arreglo
        coordenadas[k] = [ia, ja]
    
    plt.imshow(PIA)
    plt.grid(None)
    
    # Guardar la imagen en formato PDF
    plt.savefig('./datos/recorte_flev_.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    # Guardar las coordenadas en un archivo de texto
    np.savetxt('./datos/fusion_pca.txt', coordenadas, fmt='%d', delimiter='\t')
    
    plt.show() 
    return PIA






# def show_evidence5(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
#     PIA = pauli.copy()
    
#     # Coordenadas del rectángulo de recorte
#     rect_coords = [[124, 204], [364, 204], [364, 418], [124, 418]]
    
#     # Recorte de la imagen Pauli
#     xmin, ymin = np.min(rect_coords, axis=0)
#     xmax, ymax = np.max(rect_coords, axis=0)
#     PIA_cropped = PIA[ymin:ymax, xmin:xmax]
    
#     # Ajustar los puntos de borde detectados al recorte de imagen
#     MXC_cropped = MXC.copy()
#     MYC_cropped = MYC.copy()
#     MXC_cropped[np.logical_or(MXC < ymin, MXC >= ymax)] = np.nan
#     MYC_cropped[np.logical_or(MYC < xmin, MYC >= xmax)] = np.nan
    
#     fig, ax = plt.subplots(figsize=(20*img_rt, 20))
    
#     # Mostrar los puntos de borde detectados encima de la imagen recortada
#     for k in range(NUM_RAIOS):
#         ik = np.int(evidence[k, banda])
#         ia = MXC_cropped[k, ik]
#         ja = MYC_cropped[k, ik]
#         if not np.isnan(ia) and not np.isnan(ja):
#             ax.plot(ja - xmin, ia - ymin, marker='o', color="deepskyblue")
    
#     # Mostrar la imagen recortada con los puntos de borde detectados
#     ax.imshow(PIA_cropped)
#     ax.grid(None)
#     plt.show() 
    
#     # Guardar la imagen recortada con los puntos de borde detectados en formato PDF
#     fig, ax = plt.subplots(figsize=(20*img_rt, 20))
#     ax.imshow(PIA_cropped)
#     for k in range(NUM_RAIOS):
#         ik = np.int(evidence[k, banda])
#         ia = MXC_cropped[k, ik]
#         ja = MYC_cropped[k, ik]
#         if not np.isnan(ia) and not np.isnan(ja):
#             ax.plot(ja - xmin, ia - ymin, marker='o', color="deepskyblue")
#     ax.grid(None)
#     plt.savefig('./datos/recorte_pauli_con_puntos.pdf', format='pdf', dpi=300)
#     plt.show() 
#     return PIA_cropped







#-------------import numpy as np


# def show_evidence1(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
#     PIA = pauli.copy()
#     fig, ax = plt.subplots(figsize=(20*img_rt, 20))
#   #  plt.figure(figsize=(20*img_rt, 20))
    
#     # Coordenadas del rectángulo de recorte
#     rect_coords = [[124, 204], [364, 204], [364, 418], [124, 418]]
#     polygon = Polygon(rect_coords, closed=True, edgecolor='r', facecolor='none')

#     # Recorte de la imagen Pauli
#     xmin, ymin = np.min(rect_coords, axis=0)
#     xmax, ymax = np.max(rect_coords, axis=0)
#     PIA_recortada = PIA[ymin:ymax, xmin:xmax]
    
#     # Mostrar la imagen recortada
#     plt.imshow(PIA_recortada)
    
#     # Dibujar rectángulo de recorte
#     plt.gca().add_patch(polygon)
    
#     # Mostrar solo puntos en el área recortada
#     for k in range(NUM_RAIOS):
#         ik = np.int(evidence[k, banda])
#         ia = np.int(MXC[k, ik])
#         ja = np.int(MYC[k, ik])
#         if xmin <= ja < xmax and ymin <= ia < ymax:
#             plt.plot(ja - xmin, ia - ymin, marker='o', color="deepskyblue")
    
#     plt.grid(None)
    
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     plt.imshow(PIA)
#     plt.savefig('./datos/recorte_pauli_2.pdf', format='pdf', dpi=300)
#     plt.show()
    
#     return PIA_recortada


# def show_fusion_evidence(pauli, nrows, ncols, FUSION, img_rt):
#  	PIA=pauli.copy()
#  	plt.figure(figsize=(20*img_rt, 20))
#  	for i in range(nrows):
#  	    for j in range(ncols):
#  	        if(FUSION[i,j] != 0):
#  	            plt.plot(j,i, marker='o', color="deepskyblue")   
               
#  	plt.imshow(PIA)  
#  	plt.show()
#      #rect_coords = [124 204; 364 204; 364 418;124 418 ];



# def show_evidence2(pauli, NUM_RAIOS, MXC, MYC, img_rt, evidence, banda):
#     PIA = pauli.copy()
#     fig, ax = plt.subplots(figsize=(20*img_rt, 20))
     
#     # Coordenadas del rectángulo de recorte
#     vertices = [[124, 204], [364, 204], [364, 418], [124, 418]]
#     polygon = Polygon(vertices, closed=True, edgecolor='r', facecolor='none')
    
#     # Recorte de la imagen Pauli
#     xmin, ymin = min(vertices)
#     xmax, ymax = max(vertices)
#     ax.imshow(PIA[ymin:ymax, xmin:xmax])
    
#     # Dibujar rectángulo de recorte
#     ax.add_patch(polygon)
    
#     # Mostrar todos los puntos de borde estimados encima de la imagen recortada
#     for k in range(NUM_RAIOS):
#         ik = np.int(evidence[k, banda])
#         ia = np.int(MXC[k, ik])
#         ja = np.int(MYC[k, ik])
#         if xmin <= ja < xmax and ymin <= ia < ymax:
#             ax.plot(ja - xmin, ia - ymin, marker='o', color="deepskyblue")
    
#     # Configurar límites de los ejes
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     plt.imshow(PIA[ymin:ymax, xmin:xmax])
#     plt.savefig('./datos/recorte_pauli_2.pdf', format='pdf', dpi=300)
#     plt.show()
#     return PIA
## Shows the evidence fusion to simulated image
# Set evidence in a simulated image
def add_evidence_simulated(nrows, ncols, ncanal, evidencias):
    IM  = np.zeros([nrows, ncols, ncanal])
    for canal in range(ncanal):
        for k in range(nrows):
            ik = np.int(evidencias[k, canal])
            IM[ik, k, canal] = 1
    return IM
## Shows the evidence in simulated image
def show_evidence_simulated(pauli, NUM_RAIOS, img_rt, evidence, banda):
	PIA=pauli.copy()
	plt.figure(figsize=(20*img_rt, 20))
	for k in range(NUM_RAIOS):
    		ik = np.int(evidence[k, banda])
    		plt.plot(ik, k, marker='o', color="deepskyblue")
	plt.imshow(PIA)
	plt.show()






# da un resultado e-40

# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Filtrar los valores nan en fob
#     valid_indices = np.logical_not(np.isnan(fob))
#     valid_fob = fob[valid_indices]

#     # Transformar el vector de verosimilitud válido en un vector de probabilidades
#     prob = np.exp(valid_fob) / np.sum(np.exp(valid_fob))

#     # Rellenar los valores inválidos con nan
#     full_prob = np.empty_like(fob)
#     full_prob[:] = np.nan
#     full_prob[valid_indices] = prob

#     # Calcular la entropía de Shannon
#     entropy = stats.entropy(prob)

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datosssse.txt', 'w') as file:
#         for i in range(1, N):
#             file.write(f"{pix[i]}, {fob[i]}, {full_prob[i]}\n")

#     print("Entropía de Shannon:", entropy)


# no funciona

# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(N):
#         pix[j] = j + 1
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Filtrar los valores NaN y ceros
#     valid_fob = fob[~np.isnan(fob)]
#     valid_fob = valid_fob[valid_fob != 0]

#     # Transformar el vector de verosimilitud en un vector de probabilidades
#     prob = np.exp(valid_fob) / np.sum(np.exp(valid_fob))

#     # Calcular la entropía de Shannon
#     entropy = stats.entropy(prob)

#     plt.plot(pix, fob)
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_p.txt', 'w') as file:
#         for i in range(len(valid_fob)):
#             file.write(f"{pix[i]}, {valid_fob[i]}, {prob[i]}\n")

#     print("Entropía de Shannon:", entropy)




#no funciona
# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Filtrar los valores NaN y ceros
#     valid_fob = fob[~np.isnan(fob)]
#     valid_fob = valid_fob[valid_fob != 0]

#     # Normalizar el vector de probabilidad
#     prob = valid_fob / np.sum(valid_fob)
#     # Normalizar el vector de probabilidad entre cero y uno
#     prob_norm = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))

#     # Calcular la entropía de Shannon
#     entropy = -np.sum(prob_norm * np.log2(prob_norm))

    

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_norm2.txt', 'w') as file:
#         for i in range(len(valid_fob)):
#             x = pix[i+1]
#             y = valid_fob[i]
#             p = prob_norm[i]
#             file.write(f"{x}, {y}, {p}\n")

#     print("Entropía de Shannon:", entropy)

# dio 0.98
# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Filtrar los valores NaN y ceros
#     valid_fob = fob[~np.isnan(fob)]
#     valid_fob = valid_fob[valid_fob != 0]

#     # Normalizar los valores de verosimilitud
#     prob = valid_fob / np.sum(valid_fob)

#     # Calcular la entropía de Shannon
#     entropy = -np.sum(prob * np.log2(prob))

#     # Normalizar la entropía entre cero y uno
#     entropy_min = 0.0
#     entropy_max = np.log2(len(prob))
#     entropy_norm = (entropy - entropy_min) / (entropy_max - entropy_min)

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_normal.txt', 'w') as file:
#         for i in range(len(valid_fob)):
#             x = pix[i+1]
#             y = valid_fob[i]
#             p = prob[i]
#             file.write(f"{x}, {y}, {p}\n")

#     print("Entropía de Shannon (normalizada):", entropy_norm)



# solo utiliza fob y da 6.77
# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Filtrar los valores NaN y ceros
#     valid_fob = fob[~np.isnan(fob)]
#     valid_fob = valid_fob[valid_fob != 0]

#     # Transformar las verosimilitudes en un vector de probabilidad discreto
#     prob = valid_fob / np.sum(valid_fob)

#     # Calcular la entropía de Shannon
#     entropy = -np.sum(prob * np.log2(prob))

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_fin.txt', 'w') as file:
#         for i in range(len(valid_fob)):
#             x = pix[i+1]
#             y = valid_fob[i]
#             p = prob[i]
#             file.write(f"{x}, {y}, {p}\n")

#     print("Entropía de Shannon:", entropy)



# no funciona da 6.7
# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Obtener las coordenadas (x, y) correspondientes a pix y fob
#     coords = np.column_stack((pix[1:N], fob[1:N]))

#     # Filtrar los valores NaN y ceros
#     valid_coords = coords[~np.isnan(coords).any(axis=1)]
#     valid_coords = valid_coords[valid_coords[:, 1] != 0]

#     # Transformar las verosimilitudes en un vector de probabilidad discreto
#     verosimilitudes = valid_coords[:, 1]
#     prob = verosimilitudes / np.sum(verosimilitudes)

#     # Calcular la entropía de Shannon
#     entropy = -np.sum(prob * np.log2(prob))

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_casi.txt', 'w') as file:
#         for i in range(len(valid_coords)):
#             x = valid_coords[i][0]
#             y = valid_coords[i][1]
#             p = prob[i]
#             file.write(f"{x}, {y}, {p}\n")

#     print("Entropía de Shannon:", entropy)





# matriz de frecuencia no funciona

# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Obtener las coordenadas (x, y) correspondientes a pix y fob
#     coords = np.column_stack((pix[1:N], fob[1:N]))

#     # Filtrar los valores NaN en las coordenadas
#     valid_coords = coords[~np.isnan(coords).any(axis=1)]

#     # Crear la matriz de frecuencia
#     max_x = int(np.max(valid_coords[:, 0])) + 1
#     max_y = int(np.max(valid_coords[:, 1])) + 1
#     freq_matrix = np.zeros((max_x, max_y), dtype=int)

#     for coord in valid_coords:
#         x = int(coord[0])
#         y = int(coord[1])
#         freq_matrix[x, y] += 1

#     # Calcular el vector de probabilidades discreto
#     prob = freq_matrix / np.sum(freq_matrix)

#     # Calcular la entropía de Shannon
#     prob_nonzero = prob[prob != 0]  # Filtrar los valores cero
#     entropy_val = -np.sum(prob_nonzero * np.log2(prob_nonzero))

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_nn.txt', 'w') as file:
#         for i in range(len(valid_coords)):
#             x = valid_coords[i][0]
#             y = valid_coords[i][1]
#             p = prob[int(x), int(y)]
#             file.write(f"{x}, {y}, {p}\n")

#     print("Entropía de Shannon:", entropy_val)


# no funciono dio valres  1.8 e-39
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Obtener las coordenadas (x, y) correspondientes a pix y fob
#     coords = np.column_stack((pix[1:N], fob[1:N]))

#     # Filtrar los valores NaN y cero en coords
#     valid_coords = coords[~np.isnan(coords).any(axis=1)]
#     valid_coords = valid_coords[valid_coords[:, 1] != 0]

#     # Calcular las probabilidades utilizando una función sigmoide
#     prob = sigmoid(valid_coords[:, 1])

#     # Calcular la entropía de Shannon
#     entropy_val = -np.sum(prob * np.log2(prob))

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_sigmo2.txt', 'w') as file:
#         for i in range(len(valid_coords)):
#             x = valid_coords[i][0]
#             y = valid_coords[i][1]
#             p = prob[i]
#             file.write(f"{x}, {y}, {p}\n")

#     print("Entropía de Shannon:", entropy_val)




# este funciona bien, pero me da error en la segunda entropia porque hay ceros parece
#--------------------------------------------
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)

#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Obtener las coordenadas (x, y) correspondientes a pix y fob
#     coords = np.column_stack((pix[1:N], fob[1:N]))

#     # Filtrar los valores NaN en coords
#     valid_coords = coords[~np.isnan(coords).any(axis=1)]
    
    

#     # Calcular las probabilidades utilizando una función sigmoide
#     prob = sigmoid(valid_coords)
# # Calcular el vector de probabilidades utilizando softmax
#     # prob = softmax(valid_coords)
#     # Calcular la entropía de Shannon
#     entropy_val = -np.sum(prob * np.log2(prob))

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_sigmo.txt', 'w') as file:
#         for i in range(len(valid_coords)):
#             x = valid_coords[i][0]
#             y = valid_coords[i][1]
#             p = prob[i]
#             file.write(f"{x}, {y}, {p}\n")

#     print("Entropía de Shannon:", entropy_val)


#---------------------------------------------


# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)
#     coords = []  # Lista para almacenar las coordenadas (x, y)
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
#         x = pix[j]  # Coordenada x correspondiente a pix[j]
#         y = fob[j]  # Coordenada y correspondiente a fob[j]
#         coords.append((x, y))  # Agregar las coordenadas a la lista

#     valid_coords = []  # Lista para almacenar las coordenadas válidas (sin valores nan)
#     for coord in coords:
#         if not np.isnan(coord[1]):
#             valid_coords.append(coord)

#     # Obtener las coordenadas x e y de los valores válidos
#     valid_x = np.array([coord[0] for coord in valid_coords])
#     valid_y = np.array([coord[1] for coord in valid_coords])

#     # Transformar el vector de verosimilitud válido en un vector de probabilidades
#     prob = np.exp(valid_y) / np.sum(np.exp(valid_y))

#     # Calcular la entropía de Shannon
#     entropy = stats.entropy(prob)

#     plt.plot(valid_x, valid_y)
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_coor.txt', 'w') as file:
#         for i, coord in enumerate(valid_coords):
#             file.write(f"{coord[0]}, {coord[1]}, {prob[i]}\n")

#     print("Entropía de Shannon:", entropy)




# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)

#     # Filtrar los valores nan en fob
#     valid_indices = np.logical_not(np.isnan(fob))
#     valid_fob = fob[valid_indices]

#     # Transformar el vector de verosimilitud válido en un vector de probabilidades
#     prob = np.exp(valid_fob) / np.sum(np.exp(valid_fob))

#     # Rellenar los valores inválidos con nan
#     full_prob = np.empty_like(fob)
#     full_prob[:] = np.nan
#     full_prob[valid_indices] = prob

#     # Calcular la entropía de Shannon
#     entropy = stats.entropy(prob)

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_sin_nan.txt', 'w') as file:
#         for i in range(1, N):
#             file.write(f"{pix[i]}, {fob[i]}, {full_prob[i]}\n")

#     print("Entropía de Shannon:", entropy)



#nada
# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)
#     pro_dis = np.zeros(N)  # Vector de probabilidad discreta
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
    
#     # Paso 2: Obtener la función de densidad de probabilidad (PDF)
#     pdf = np.exp(fob)
    
#     # Paso 3: Normalizar la función de densidad de probabilidad
#     pdf_normalized = pdf / np.sum(pdf)
    
#     # Paso 4: Obtener la función de probabilidad discreta
#     pro_dis[1:N] = pdf_normalized[1:N]
    
#     plt.plot(pix[1:N], pro_dis[1:N])
#     plt.show()

#------------------------Esta está bien-----------
#--------------------------------------------------

# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('seaborn-whitegrid')
#     #plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)
#     pro_dis = np.zeros(N)  # Vector de probabilidad discreta
    
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
    
    
#     # Filtrar valores nan y ceros en fob
#     valid_indices = np.nonzero(~np.isnan(fob) & (fob != 0))
#     fob_filtered = fob[valid_indices]
    
#     # Paso 2: Obtener la función de densidad de probabilidad (PDF)
#     pdf = np.exp(fob_filtered)
    
#     # Paso 3: Normalizar la función de densidad de probabilidad
#     pdf_normalized = pdf / np.sum(pdf)
    
#     # Paso 4: Obtener la función de probabilidad discreta
#     pro_dis[valid_indices] = pdf_normalized
    
#     # Calcular la entropía de Shannon
#     entropy = -np.sum(pro_dis[valid_indices] * np.log2(pro_dis[valid_indices]))
    
#     # Guardar los resultados de pix[j] y fob[j] en un archivo de texto
#     np.savetxt('./datos/resultados_nuevos.txt', np.column_stack((pix[valid_indices], fob_filtered)), fmt='%.5f', delimiter='\t')
    
#     plt.plot(pix[valid_indices], pro_dis[valid_indices])
#     plt.show()
    
#     # Mostrar el resultado de la entropía en pantalla
#     print("Entropía de Shannon:", entropy)

#     # Aplicar transformación a los valores del eje y
#     fob_transformed = fob - min(fob)  # Restar el valor mínimo para que comience desde cero
    
#     plt.plot(pix[1:N], fob_transformed[1:N])
    
#     # Configurar ejes y leyendas
#     plt.xlabel('Pixel j')
#     plt.ylabel('L(j)')
#     # plt.title('Total Likelihood Function')
    

#     # Guardar la imagen en formato PDF
#     plt.savefig('./datos/total_likelihood_1.pdf', format='pdf', dpi=300)
    
#     plt.show() 
    
    #------------------------Esta está bien-----------
    #--------------------------------------------------
    
def plot_total_likelihood(z, N, matdf1, matdf2, rayo_index):
    plt.style.use('seaborn-whitegrid')
    #plt.style.use('ggplot')
    pix = np.zeros(N)
    fob = np.zeros(N)
    pro_dis = np.zeros(N)  # Vector de probabilidad discreta
    
    for j in range(1, N):
        pix[j] = j
        fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
    
    
    # Filtrar valores nan y ceros en fob
    valid_indices = np.nonzero(~np.isnan(fob) & (fob != 0))
    fob_filtered = fob[valid_indices]
    
    # Paso 2: Obtener la función de densidad de probabilidad (PDF)
    pdf = np.exp(fob_filtered)
    
    # Paso 3: Normalizar la función de densidad de probabilidad
    pdf_normalized = pdf / np.sum(pdf)
    
    # Paso 4: Obtener la función de probabilidad discreta
    pro_dis[valid_indices] = pdf_normalized
    
    # Calcular la entropía de Shannon
    entropy = -np.sum(pro_dis[valid_indices] * np.log2(pro_dis[valid_indices]))
    
    # Guardar los resultados de pix[j] y fob[j] en un archivo de texto
    #np.savetxt('./datos/resultados_nuevos.txt', np.column_stack((pix[valid_indices], fob_filtered)), fmt='%.5f', delimiter='\t')
    #grafica la probabilidad
    #plt.plot(pix[valid_indices], pro_dis[valid_indices])
    #plt.show()
    
## para graficar las verosimilanzas
    # Aplicar transformación a los valores del eje y
    fob_transformed = fob - min(fob)  # Restar el valor mínimo para que comience desde cero
    
    plt.plot(pix[1:N], fob_transformed[1:N], color="red")
    #plt.plot(pix[1:N], fob[1:N])
    
    # Configurar ejes y leyendas
    plt.xlabel('Pixel j')
    plt.ylabel('L(j)')
    # plt.title('Total Likelihood Function')
    

    # Guardar la imagen en formato PDF
    #plt.savefig('./datos/total_likelihood_1.pdf', format='pdf', dpi=300)
    # Guardar la imagen en formato PDF
    plt.savefig(f'./datos/likelihood_{rayo_index}.pdf', format='pdf', dpi=300)
    
    
    
    plt.show() 
    
    return entropy
    
    
#--------------ORIGINAL

# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('ggplot')
#     pix = np.zeros(N)
#     fob = np.zeros(N)
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
#     #plt.plot(pix[1:N], fob[1:N])
#     #plt.show()
#   #-------------  
#     # Guardar los valores en un archivo de texto
#     #with open('./datos/datos_like.txt', 'w') as file:
#         #for i in range(1, N):
#             #file.write(f"{pix[i]}, {fob[i]}\n")
#          # Normalizar los valores en fob
#     # Normalizar los valores en fob
#     fob -= scipy.special.logsumexp(fob)

#     # Transformar el vector de verosimilitud en un vector de probabilidades
#     prob = np.exp(fob)

#     # Calcular la entropía de Shannon
#     entropy = stats.entropy(prob)

#     plt.plot(pix[1:N], fob[1:N])
#     plt.show()

#     # Guardar los valores en un archivo de texto
#     with open('./datos/datos_like3.txt', 'w') as file:
#         for i in range(1, N):
#             file.write(f"{pix[i]}, {fob[i]}, {prob[i]}\n")

#     print("Entropía de Shannon:", entropy)
    # fob_normalized = fob - np.max(fob)  # Restar el valor máximo para evitar problemas de desbordamiento

    # # Transformar el vector de verosimilitud normalizado en un vector de probabilidades
    # prob = np.exp(fob_normalized)/ np.sum(np.exp(fob_normalized))       
            
    #  # Transformar el vector de verosimilitud en un vector de probabilidades
    # #prob = np.exp(fob) / np.sum(np.exp(fob))

    # # Calcular la entropía de Shannon
    # entropy = stats.entropy(prob)

    # plt.plot(pix[1:N], fob[1:N])
    # plt.show()

    # # Guardar los valores en un archivo de texto
    # with open('./datos/datoslike2.txt', 'w') as file:
    #     for i in range(1, N):
    #         file.write(f"{pix[i]}, {fob[i]}, {prob[i]}\n")

    # print("Entropía de Shannon:", entropy)            
    
#MEJORAR LA GRAFICA    
# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('seaborn-whitegrid')  # Otro estilo de gráfico
    
#     pix = np.zeros(N)
#     fob = np.zeros(N)
    
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
    
#     plt.plot(pix[1:N], fob[1:N])
    
#     # Configurar ejes y leyendas
#     plt.xlabel('Pixel j')  # Etiqueta del eje x
#     #plt.ylabel('Function Objective (-)')  # Etiqueta del eje y
#     #plt.title('Total Likelihood Function')  # Título del gráfico


# def plot_total_likelihood(z, N, matdf1, matdf2):
#     plt.style.use('seaborn-whitegrid')
    
#     pix = np.zeros(N)
#     fob = np.zeros(N)
    
#     for j in range(1, N):
#         pix[j] = j
#         fob[j] = -ptl.func_obj_l_L_mu(pix[j], z, N, matdf1, matdf2)
    
#     # Aplicar transformación a los valores del eje y
#     fob_transformed = fob - min(fob)  # Restar el valor mínimo para que comience desde cero
    
#     plt.plot(pix[1:N], fob_transformed[1:N])
    
#     # Configurar ejes y leyendas
#     plt.xlabel('Pixel j')
#     plt.ylabel('L(j)')
#    # plt.title('Total Likelihood Function')
    

#     # Guardar la imagen en formato PDF
#     plt.savefig('./datos/total_likelihood_1.pdf', format='pdf', dpi=300)
    
#     plt.show()   
#
def show_image(IMG, nrows, ncols, img_rt):
    plt.figure(figsize=(20*img_rt, 20))
    escale = np.mean(IMG) * 2
    plt.imshow(IMG,clim=(0.0, escale), cmap="gray")
    plt.show()
#
def show_image_pauli(IMG, nrows, ncols, img_rt):
    plt.figure(figsize=(20*img_rt, 20))
    plt.imshow(IMG)
    plt.show()
#
def show_image_to_file(IMG, nrows, ncols, image_name):
    directory = 'figure'
    image = str(image_name) + '.pdf'
    file_path = os.path.join(directory, image)
    escale = np.mean(IMG) * 2
    plt.imsave(file_path, IMG, cmap="gray", vmin = 0, vmax = escale)
    #
def show_image_pauli_to_file(IMG, nrows, ncols, image_name):
    #cwd = os.getcwd()
    directory = './figure/'
    print(directory)
    image = str(image_name) + '.pdf'
    print(directory)
    #file_path = os.path.join(cwd, directory, image)
    file_path = os.path.join(directory, image)
    print(file_path)
    print("#########")
    print(np.max(IMG))
    print(np.min(IMG))
    IMG = IMG/5
    #plt.imshow(IMG)
    #plt.show()
    plt.imsave(file_path, IMG)
    #plt.imsave(C:/Users/ander/OneDrive/Documentos/MEGA/MEGAsync/mack/alejandro/2023/ArtRemSens01/ArtigoFonte/figure/FlevEvChhRoi01.pdf, IMG)
    
#
def show_image_perfil_h(img, cont, channel):
    plt.style.use('ggplot')
    plt.plot(img[:, cont, channel])
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.show()
#
def fun_smooth_ramp_3d_corner(x, y):
    f1 = 6.0 * x**5 - 15.0 * x**4 + 10 * x**3
    f2 = 6.0 * y**5 - 15.0 * y**4 + 10 * y**3
    f = f1 * f2
    return f
#
def fun_smooth_ramp(x):
    f = 6.0 * x**5 - 15.0 * x**4 + 10 * x**3
    return f
#
def fun_smooth_ramp_3d_set(x, y, nrows, ncols):
    eps = 50;
    sx = 100;
    sy = 200;
    xl = int(nrows/2 - sx);
    xu = int(nrows/2 + sx);
    yl = int(ncols/2 - sy);
    yu = int(ncols/2 + sy);
    z = np.zeros((nrows, ncols))
    # Build a rectangle [xl + eps, xu - eps] X [yl + eps, yu - eps] 
    for i in range(xl + eps, xu - eps):
        for j in range(yl + eps, yu - eps):
            z[i, j] = 1.0  
    # Upper rectangle [xl - eps, xl + eps] X [yl + eps, yu - eps]
    for i in range(xl - eps, xl + eps):
        for j in range(yl + eps, yu - eps):
            xaux = (i - (xl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp(xaux)  
    # Lower rectangle  [xu - eps, xu + eps] X [yl + eps, yu - eps]
    for i in range(xu - eps, xu + eps):
        for j in range(yl + eps, yu - eps):
            xaux = (i - (xu - eps)) / (2 * eps)
            z[i, j] = 1.0 - fun_smooth_ramp(xaux)  
    # left rectangle [xl + eps, xu - eps] X [yl - eps, yl + eps]
    for i in range(xl + eps, xu - eps):
        for j in range(yl - eps, yl + eps):
            yaux = (j - (yl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp(yaux)  
    # right rectangle [xl + eps, xu - eps] X [yu - eps, yu + eps]
    for i in range(xl + eps, xu - eps):
        for j in range(yu - eps, yu + eps):
            yaux = (j - (yu - eps)) / (2 * eps)
            z[i, j] = 1.0 - fun_smooth_ramp(yaux)
    # Smoth ramp to corner
    # [xl - eps, xl + eps] X [yl - eps, yl + eps]
    for i in range(xl - eps, xl + eps):
        for j in range(yl - eps, yl + eps):
            xaux = (i - (xl - eps)) / (2 * eps)
            yaux = (j - (yl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)  
    # [xl - eps, xl + eps] X [yu - eps, yu + eps]
    for i in range(xl - eps, xl + eps):
        for j in range(yu - eps, yu + eps):
            xaux =       (i - (xl - eps)) / (2 * eps)
            yaux = 1.0 - (j - (yu - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)  
    # [xu - eps, xu + eps] X [yl - eps, yl + eps]
    for i in range(xu - eps, xu + eps):
        for j in range(yl - eps, yl + eps):
            xaux = 1.0 - (i - (xu - eps)) / (2 * eps)
            yaux =       (j - (yl - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)  
    # [xu - eps, xu + eps] X [yu - eps, yu + eps]
    for i in range(xu - eps, xu + eps):
        for j in range(yu - eps, yu + eps):
            xaux =   1.0 -  (i - (xu - eps)) / (2 * eps)
            yaux =   1.0 -  (j - (yu - eps)) / (2 * eps)
            z[i, j] = fun_smooth_ramp_3d_corner(xaux, yaux)  
    return z
def plot_3d_edge(nrows, ncols, image_name):
    #directory = './figuras/'
    #figure = str(figure_name) + '.pdf'
    #file_path = os.path.join(directory, figure)
    # Domain [a, b] x [c, d]
    #a = 0
    #b = nrows
    #
    #c = 0
    #d = ncols
    #
    x = np.arange(0, nrows, 1)
    y = np.arange(0, ncols, 1)
    x, y = np.meshgrid(x, y)
    #z = fun_smooth_ramp_3d_set(x, y, nrows, ncols) 
    #fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=True)
    #surf = ax.plot_surface(x, y, z,
    #					rstride = 1,
    # 					cstride = 1,
    #					alpha = 0.4,
    #					color= 'lightgreen')
    #  Gráfico reta 1
    #x1 = np.linspace(0, 4, 100)
    #y1 = x1
    #ax.plot(x1, y1, zs=0, zdir='z', color='blue')
    #  Gráfico reta 2
    #x1 = np.linspace(0, 3, 100)
    #y1 = x1 + np.sqrt(2)
    #ax.plot(x1, y1, zs=0, zdir='z', color='blue')
    #  Plot reta 3
    #x1 = np.linspace(0, 4, 100)
    #y1 = -x1 + 4
    #ax.plot(x1, y1, zs=0, zdir='z', color='orange')
    #z2 = 1
    #x2 = np.linspace(a - np.sqrt(z2), a + np.sqrt(z2), 100)
    #y2 =  np.sqrt(z2 - (x2 - a)**2) + b
    #y3 = -np.sqrt(z2 - (x2 - a)**2) + b
    #ax.plot(x2, y2, zs=0, zdir='z', color='blue')
    #ax.plot(x2, y3, zs=0, zdir='z', color='blue')
    #
    #ax.plot(x2, y2, zs=1, zdir='z', color='green')
    #ax.plot(x2, y3, zs=1, zdir='z', color='green')
    # plot curva de nível 2
    #z2 = 2
    #x2 = np.linspace(a - np.sqrt(z2) + epsilon, a + np.sqrt(z2), 100)
    #y2 =  np.sqrt(z2 - (x2 - a)**2) + b
    #y3 = -np.sqrt(z2 - (x2 - a)**2) + b
    #ax.plot(x2, y2, zs=0, zdir='z', color='blue')
    #ax.plot(x2, y3, zs=0, zdir='z', color='blue')
    #
    #ax.plot(x2, y2, zs=2, zdir='z', color='green')
    #ax.plot(x2, y3, zs=2, zdir='z', color='green')
    # plot curva de nível 3
    #z2 = 3
    #x2 = np.linspace(a - np.sqrt(z2) + epsilon, a + np.sqrt(z2), 100)
    #y2 =  np.sqrt(z2 - (x2 - a)**2) + b
    #y3 = -np.sqrt(z2 - (x2 - a)**2) + b
    #ax.plot(x2, y2, zs=0, zdir='z', color='blue')
    #ax.plot(x2, y3, zs=0, zdir='z', color='blue')
    #
    #ax.plot(x2, y2, zs=3, zdir='z', color='green')
    #ax.plot(x2, y3, zs=3, zdir='z', color='green')
    #
    #theta = np.linspace(0.6, 2.0, 100)
    #xc = theta
    #yc = xc + np.sqrt(2)
    #zc = (xc - a)**2 + (yc - b)**2
    #ax.plot(xc, yc, zc, color='red')
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    plt.xlabel('x Pixel')
    ax.set_ylabel('y Pixel')
    #ax.set_xlim(0, 4)
    #ax.set_ylim(0, 4)
    ax.set_zlabel('Pixel intensity')
    ax.set_title('Function smooth ramp')
    plt.show()

#
