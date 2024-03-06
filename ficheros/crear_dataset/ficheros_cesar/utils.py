# funciones se usan en varios sitios
# 
import glob
import numpy as np
import imageio
import scipy.stats as stats
import cv2


#Blending modes
def blendStack(stack, modo='median', axis=0):
    if modo == 'sum':
        blend = np.sum(stack, axis)
        
    if modo == 'arithmetic mean':
        blend = np.mean(stack, axis)
    
    if modo == 'geometric mean':
        blend = stats.gmean(stack, axis)
    
    if modo == 'harmonic mean':
        blend = stats.hmean(stack, axis)
    
    if modo == 'median':
        blend = np.median(stack, axis)
    
    if modo == 'minimum':
        blend = np.amin(stack, axis)

    if modo == 'maximum':
        blend = np.amax(stack, axis)

    if modo == 'curtosis':
        blend = stats.kurtosis(stack, axis)

    if modo == 'variance':
        blend = np.var(stack, axis)

    if modo == 'standard deviation':
        blend = np.std(stack, axis)

    return blend.astype(stack.dtype)


#Blending modes, devuelve para cada píxel, qué imagen tiene el valor máximo (o mediana)
def blendStack2(stack, modo='median', axis=0):
    if modo == 'sum':
        blend = np.sum(stack, axis)
        
    if modo == 'arithmetic mean':
        blend = np.mean(stack, axis)
    
    if modo == 'geometric mean':
        blend = stats.gmean(stack, axis)
    
    if modo == 'harmonic mean':
        blend = stats.hmean(stack, axis)
    
    if modo == 'median':
        blend = np.median(stack, axis)
        labels_blend = stack.argmax(axis=0)
    
    if modo == 'minimum':
        blend = np.amin(stack, axis)
        labels_blend = stack.argmax(axis=0)

    if modo == 'maximum':
        blend = np.amax(stack, axis)
        labels_blend = stack.argmax(axis=0)

    if modo == 'curtosis':
        blend = stats.kurtosis(stack, axis)

    if modo == 'variance':
        blend = np.var(stack, axis)

    if modo == 'standard deviation':
        blend = np.std(stack, axis)

    return blend.astype(stack.dtype), labels_blend.astype(np.uint16)

# leer todas las fotos en un directorio por fecha del archivo, creando un array [numfoto, y,x,c]

def stackRead(pathname):
    '''
    pathname defined by "glob" pattern.
    i.e.: "directory/sequence_folder/image_*.jpg"
    '''
    # List of image in pathname folder
    SEQ_IMG = glob.glob(pathname)
    n = len(SEQ_IMG)
    # sample for stack definition
    sample = imageio.imread(SEQ_IMG[0])
    # x and y are the dimensions
    # c is the number of channels
    y, x, c = sample.shape
    # define stack
    stack = np.zeros((n, y, x, c), dtype=sample.dtype)
    # image stacking
    for FILE in SEQ_IMG:
        index = SEQ_IMG.index(FILE)
        stack[index] = imageio.imread(FILE)
    # output
    return stack


#   NO FUNCIONA BIEN, NO USAR
# máscara para el fondo (cielo con estrellas). Píxeles que valen 255 son del background (cielo); cero son del foregound (suelo)
# es el algoritmo 2 de 0e_getbackground.ipynb
# inputs: fusiona_sin_artefacto: array 3D resultado de fusionar todas las fotos sin mediana ni artefactos (es decir, circumpolar sin mediana)
# output: máscara: 0 suelo, 255 cielo
def getbackground(fusiona_sin_artefacto):
    y, x, c = fusiona_sin_artefacto.shape
    mask = 255*np.ones((y, x), dtype='uint8') # inicializamos a todo es cielo
    graycircump_sinmedian = cv2.cvtColor(fusiona_sin_artefacto, cv2.COLOR_BGR2GRAY) 
    (T, thresh1) = cv2.threshold(graycircump_sinmedian, 15, 255,cv2.THRESH_BINARY) # binarizar
    kernel = np.ones((3,3),np.uint8)
    graycircump_sinmedian_limpio = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)     #limpiar píxeles blancos ruido
    # cada  columna buscar el último píxel blanco (estrella)
    for i in range(x):
        col = graycircump_sinmedian_limpio[:,i]
        result = np.where(col == 255)
        sol=result[0]
        if sol.size != 0:
            p=sol.max()
            #print(p)
            mask[p:,i]=0
    # depurar máscara
    kernel = np.ones((1,10),np.uint8) # rellenar de negro columnas que no ha detectado estrellas
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask1




#   no se si FUNCIONA BIEN, tiene buena pinta
# máscara para el fondo (cielo con estrellas). Píxeles que valen 255 son del background (cielo); cero son del foregound (suelo)
# es el algoritmo 0 de 0e_getbackground.ipynb
# inputs: la mediana (formato RGB)
# output: máscara: 0 suelo, 255 cielo
def getbackground0(mediana):
    y, x, c = mediana.shape
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
    #rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(mediana) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    bordestmp = 255*edges
    bordes = bordestmp.astype('uint8')
    imageio.imwrite('b0.jpg',bordes, quality=100)            

    # contornos
    ret, thresh = cv2.threshold(bordes, 20, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((y, x), dtype='uint8') # inicializamos a todo es suelo
    # la mediana solo tiene la estrella polar, así que los bordes deberían ser entre foreground y background
    cv2.drawContours(mask, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    imageio.imwrite('b1.jpg',mask, quality=100)            

    # unir bordes por si acaso
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.dilate(mask,kernel,iterations = 3)
    imageio.imwrite('b2.jpg',mask, quality=100)            

    output = dst.copy()
    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(dst)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
    sizes = sizes[1:]
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    min_size = 1500  

    # output image with only the kept components
    im_result = np.zeros((output.shape))
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255

    imageio.imwrite('b3.jpg',im_result, quality=100)            

    # imagen limpia, crea máscara 

    y, x, c = mediana.shape
    lim = y/5 # como mucho de una columna a la siguiente el horizonte se va un 20% de la altura de la foto: 1000 píxeles, pues 100; si se va más se modifica por el valor columna anterior
    mask = 255*np.ones((y, x), dtype='uint8') # inicializamos a todo es cielo
    # primera columna
    col = im_result[:,0]
    result = np.where(col == 255)
    sol=result[0]
    if sol.size != 0:
        p=sol.min()
        mask[p:,0]=0
        p_ant = p
    else: p_ant = round(2*y/3) # aleatorio regla tercios o mejor podía poner la media de las columnas (quitando las columnas sin suelo)
    #resto columnas
    for i in range(1,x):
        col = im_result[:,i]
        result = np.where(col == 255)
        sol=result[0]
        if sol.size != 0:
            p=sol.min()
            if (np.abs(p - p_ant) > lim  and i!=1): 
                p = p_ant   # mejora: interpolar entre los extremos, en vez de copiar anterior (valor plano)
                #print(p)
        else:
            p = p_ant
        mask[p:,i]=0
        p_ant = p  # actualizar para siguiente columna
    imageio.imwrite('b4.jpg',mask, quality=100)            

    return mask


#####################
# Detector de bordes (mejor que Canny)
# Output: bordes en uint8 (no binario)
def bordes(img):
    y, x, c = img.shape
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
    #rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(img) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    bordestmp = 255*edges
    bordes = bordestmp.astype('uint8')
    #imageio.imwrite('b0.jpg',bordes, quality=100)            
    return bordes
