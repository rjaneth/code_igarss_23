### Version 03/08/2021
# Article GRSL
# Ref:
# A. A. De Borba,
# M. Marengoni and
# A. C. Frery,
# "Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images,"
# in IEEE Geoscience and Remote Sensing Letters,
#doi: 10.1109/LGRS.2020.3022511.
# bibtex
#@ARTICLE{9203845,
#  author={De Borba, Anderson A. and Marengoni, Maurício and Frery, Alejandro C.},
#  journal={IEEE Geoscience and Remote Sensing Letters},
#  title={Fusion of Evidences in Intensity Channels for Edge Detection in PolSAR Images},
#  year={2020},
#  volume={},
#  number={},
#  pages={1-5},
#  doi={10.1109/LGRS.2020.3022511}}
#
## Import all required libraries
import numpy as np
## Used to read images in the mat format
import scipy.io as sio
## Used to equalize histograms in images
from skimage import exposure
## Used to present the images
import matplotlib as mpl
import matplotlib.pyplot as plt
## Used to find border evidences
import math
from scipy.optimize import dual_annealing
## Used in the DWT and SWT fusion methods
import pywt
#### Used to find_evidence_bfgs
from scipy.optimize import minimize
## Used
### Import mod
# see file  /Misc/mod_code_py.pdf
#
#import os
import polsar_basics as pb
import polsar_loglikelihood as plk
import polsar_fusion as pf
import polsar_total_loglikelihood as ptl
import polsar_evidence_lib as pel
import polsar_plot as pplt

# Crear la carpeta "data" si no existe
#if not os.path.exists('datos'):
   # os.makedirs('datos')
#
## This function defines the source image and all the dat related to the region where we want
## to find borders
## Defines the ROI center and the ROI boundaries. The ROI is always a quadrilateral defined from the top left corner
## in a clockwise direction.
#
def select_data():
    print("Select the image to be processed:")
    print("1.Flevoland - area 1")
    print("2.San Francisco")
    opcao=int(input("type the option:"))
    if opcao==1:
        print("Computing Flevoland area - region 1")
        imagem="./Data/AirSAR_Flevoland_Enxuto.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=278
        dy=69##64
        ## ROI coordinates
        x1 = 157;
        y1 = 284;
        x2 = 309;
        y2 = 281;
        x3 = 310;
        y3 = 327;
        x4 = 157;
        y4 = 330;
        ## inicial angle to start generating the radius
        alpha_i=0.0
        ## final angle to start generating the radius
        alpha_f =  2 * np.pi
        ## slack constant
        lim = 14
    else:
        print("Computing San Francisco Bay area - region 1")
        imagem="./Data/SanFrancisco_Bay.mat"
        ## values adjusted visually - it needs to be defined more preciselly
        ## delta values from the image center to the ROI center
        dx=50
        dy=-195
        ## ROI coordinates
        x1 = 180;
        y1 = 362;
        x2 = 244;
        y2 = 354;
        x3 = 250;
        y3 = 420;
        x4 = 188;
        y4 = 427;
        ## inicial angle to start generating the radius
        alpha_i= np.pi
        ## final angle to start generating the radius
        alpha_f = 3 * np.pi / 2
        ## slack constant
        lim = 20
    ## Radius length
    RAIO=120
    ## Number of radius used to find evidence considering a whole circunference
    NUM_RAIOS=50
    ## adjust the number of radius based on the angle defined above
    if (alpha_f-alpha_i)!=(2*np.pi):
        NUM_RAIOS=int(NUM_RAIOS*(alpha_f-alpha_i)/(2*np.pi))
    gt_coords=[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    #
    return imagem, dx, dy, RAIO, NUM_RAIOS, alpha_i, alpha_f, lim, gt_coords
# ### A célula abaixo funciona como um main do código de fusão de evidências de borda em imagens POLSAR - ainda deverá ser editado para uma melhor compreensão do código ###
# The code works as main to GRSL2020 codes
#
cs1 = 'FlevEvChhRoi01Span'
cs2 = 'FlevEvChhRoi02Span'
cs3 = 'FlevEvChhRoi03Span'
## Define the image and the data from the ROI in the image
imagem, dx, dy, RAIO, NUM_RAIOS, alpha_i, alpha_f, lim, gt_coords = select_data()
#
## Reads the image and return the image, its shape and the number of channels
img, nrows, ncols, nc = pb.le_imagem(imagem)
#
## Plot parameter
img_rt = nrows/ncols
#
## Uses the Pauli decomposition to generate a visible image
PI = pb.show_Pauli(img, 1, 0)
#
## Define the radius in the ROI
x0, y0, xr, yr = pb.define_radiais(RAIO, NUM_RAIOS, dx, dy, nrows, ncols, alpha_i, alpha_f)
#
MXC, MYC, MY, IT, PI = pb.desenha_raios(ncols, nrows, nc, RAIO, NUM_RAIOS, img, PI, x0, y0, xr, yr)
#
#results = pb.desenha_raios(ncols, nrows, nc, RAIO, NUM_RAIOS, img, PI, x0, y0, xr, yr)
#print(MXC)
## Define the number of channels to be used to find evidence
## and realize the fusion in the ROI GHJ
ncanal = 3
#total_canal = 9
evidencias = np.zeros((NUM_RAIOS, ncanal))  

#rayo_idx = 2
## Find the evidences
# Intensity channals pdf (gamma pdf)

evidencias[:, 0] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, 0, MY, lim)
evidencias[:, 1] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, 1, MY, lim)
evidencias[:, 2] = pel.find_evidence_bfgs(RAIO, NUM_RAIOS, 2, MY, lim)

#print(f"Evidencia {k + 1}: Coordenadas (x, y) = ({x}, {y})")
    
# Guardar los resultados en un archivo .txt
#np.savetxt('./datos/evidencias_pares_hv.txt', evidencias, fmt='%d', delimiter='\t')
# Guardar los resultados en archivos .txt
#np.savetxt('./datos/evidencia_00.txt', evidencias[:, 0], fmt='%0.5f')
#np.savetxt('./datos/evidencia_hv.txt', evidencias[:, 0], fmt='%0.5f')
# Guardar los resultados en un archivo .txt
#np.savetxt('./datos/evidencia_xy_hv.txt', evidencias, fmt='%.5f', delimiter='\t')
#np.savetxt('./datos/evidencia_22.txt', evidencias[:, 2], fmt='%0.5f')

#------------



#-----------
# Span pdf
#evidencias[:, 3] =  pel.find_evidence_bfgs_span(RAIO, NUM_RAIOS, MY, lim)
# Ratio intensities pdf ( inum / idem )
# 0 = HH, 1 = HV, 2 = VV
inum = 0
idem = 1
#evidencias[:, 4] = pel.find_evidence_bfgs_intensity_ratio_three_param(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 0
idem = 2
#evidencias[:, 5] = pel.find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 1
idem = 2
#evidencias[:, 6] = pel.find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 1
idem = 0
#evidencias[:, 7] = pel.find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 2
idem = 1
#evidencias[:, 8] = pel.find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, MY, lim, inum, idem)
inum = 2
idem = 0
#evidencias[:, 9] = pel.find_evidence_bfgs_intensity_ratio(RAIO, NUM_RAIOS, MY, lim, inum, idem)
#
#c1 = 0
#c2 = 1
#evidencias[:, 10] = pel.find_evidence_bfgs_int_prod_biv(RAIO, NUM_RAIOS, MY, lim, c1, c2)
mul1 = 0
mul2 = 1


## Put the evidences in an image
IM = pel.add_evidence(nrows, ncols, ncanal, evidencias, NUM_RAIOS, MXC, MYC)
#np.savetxt('./datos/IM.txt', IM, fmt='%0.5f')
## Computes fusion using mean - metodo = 1
#MEDIA = pf.fusao(IM, 1, NUM_RAIOS)

## Computes fusion using pca - metodo = 2
#PCA = pf.fusao(IM, 2, NUM_RAIOS)

## Computes fusion using ROC - metodo = 3
#ROC = pf.fusao(IM, 3, NUM_RAIOS)

## Testing fusion using SVD - metodo = 4
#SVD = pf.fusao(IM, 4, NUM_RAIOS)

## Testing fusion using SWT - metodo = 5
#SWT = pf.fusao(IM, 5, NUM_RAIOS)

## Testing fusion using SWT - metodo = 5
SWT_PCA = pf.fusao(IM, 8, NUM_RAIOS)


## Testing fusion using DWT - metodo = 6
#DWT = pf.fusao(IM, 6, NUM_RAIOS)
#

# Guardar los resultados en archivos .txt
#np.savetxt('./datos/MEDIA_2.txt', MEDIA, fmt='%0.5f')
#np.savetxt('./datos/PCA.txt', PCA, fmt='%0.5f')
#np.savetxt('./datos/ROC.txt', ROC, fmt='%0.5f')

# The edges evidence images are shown
#[3]

#lt.grid(None)
PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 0)

#pplt.show_image_pauli_to_file(PIE, nrows, ncols, cs1)
PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 1)
#pplt.show_image_pauli_to_file(PIE, nrows, ncols, cs2)
#
PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 2)
#pplt.show_image_pauli_to_file(PIE, nrows, ncols, cs3)

plt.grid(None)
#IE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 0)
#PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 4)
#PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 5)
#PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 6)
#PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 7)
#PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 8)
#PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 9)
#PIE = pplt.show_evidence(PI, NUM_RAIOS, MXC, MYC, img_rt, evidencias, 10)

#
# The images with fusion of evidence are shown
#

#pf.show_fusion_evidence(PI, nrows, ncols, MEDIA, img_rt)
#pf.show_fusion_evidence(PI, nrows, ncols, PCA, img_rt)
#pf.show_fusion_evidence(PI, nrows, ncols, ROC, img_rt)
#pf.show_fusion_evidence(PI, nrows, ncols, DWT, img_rt)
#pf.show_fusion_evidence(PI, nrows, ncols, SWT, img_rt)
#pf.show_fusion_evidence(PI, nrows, ncols, SVD, img_rt)
pf.show_fusion_evidence(PI, nrows, ncols, SWT_PCA, img_rt)


#plot total likehood
# z = np.zeros(RAIO)
# Le = 4
# Ld = 4
# evidencias = np.zeros(NUM_RAIOS)
# for k in range(NUM_RAIOS):
#     z = MY[k, :, ncanal]
#     zaux = np.zeros(RAIO)
#     conta = 0
#     for i in range(RAIO):
#         if z[i] > 0:
#             zaux[conta] = z[i]
#             conta = conta + 1
#     #
#     indx  = pb.get_indexes(zaux != 0) 
#     N = int(np.max(indx))
#     z =  zaux[1:N]
#     matdf1 =  np.zeros((N, 2))
#     matdf2 =  np.zeros((N, 2))
#     for j in range(1, N):
#         mue = sum(z[0: j]) / j
#         matdf1[j, 0] = Le
#         matdf1[j, 1] = mue
#         mud = sum(z[j: (N + 1)]) / (N - j)
#         matdf2[j, 0] = Ld
#         matdf2[j, 1] = mud
#     #
#     lw = [lim]
#     up = [N - lim]
#     #
#     pplt.plot_total_likelihood(z, N, matdf1, matdf2)





z = np.zeros(RAIO)
Le = 4
Ld = 4


evidencias = np.zeros(NUM_RAIOS)

for k in range(NUM_RAIOS):
    z = MY[k, :, ncanal]
    zaux = np.zeros(RAIO)
    conta = 0
    for i in range(RAIO):
        if z[i] > 0:
            zaux[conta] = z[i]
            conta = conta + 1
    
    indx = pb.get_indexes(zaux != 0) 
    N = int(np.max(indx))
    z =  zaux[1:N]
    matdf1 =  np.zeros((N, 2))
    matdf2 =  np.zeros((N, 2))
    
    for j in range(1, N):
        mue = sum(z[0: j]) / j
        matdf1[j, 0] = Le
        matdf1[j, 1] = mue
        mud = sum(z[j: (N + 1)]) / (N - j)
        matdf2[j, 0] = Ld
        matdf2[j, 1] = mud
    
    entropy = pplt.plot_total_likelihood(z, N, matdf1, matdf2, k+1)
    evidencias[k] = entropy

# Mostrar el resultado de la entropía para cada conjunto de datos de rayo
for i, entropy in enumerate(evidencias):
    print(f"Entropía de Shannon para el rayo {i+1}: {entropy}")