
#FICHERO CON TODAS LAS DIFERENTES FUNCIONES QUE CREAN TRAZAS FALSAS !!!!!!!!!! lo importo en todoJunto-clase.ipynb
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import math
from noise import pnoise2
import noise

#--------------------------------------------- TIPO 1---------------------------------------------------------------
#una traza a puntos rojos y blancos que se va degradando (perdiendo opacidad)

def trazas_tipo1(ruta_imagen, ruta_guardado_img,ruta_guardado_unet):
    imagen = Image.open(ruta_imagen)
    ancho, alto = imagen.size

    num_trazas = random.randint(1, 4)

    #SOLUCION UNET
    solUNET = np.zeros((alto, ancho))  # Crear una matriz de ceros del mismo tamaño que la imagen

    for _ in range(num_trazas):

        # --------- 1. coordenadas de inicio y fin
        coordenada_inicio = (random.uniform(0, ancho), random.uniform(0, alto))
        x = random.uniform(60, 400) 
        coordenada_fin = (coordenada_inicio[0] + x, coordenada_inicio[1] + x)

        # --------- 2. parametros:
        diametro_circulo = 0.05
        separacion = 8

        # --------- 3. pendiente y la longitud de la línea
        delta_x = coordenada_fin[0] - coordenada_inicio[0]
        delta_y = coordenada_fin[1] - coordenada_inicio[1]
        longitud = np.sqrt(delta_x**2 + delta_y**2)
        pendiente = delta_y / delta_x if delta_x != 0 else float('inf')

        # --------- 4. elipses a lo largo de la línea con colores alternados entre rojo y blanco
        coordenadas = []
        colores = []
        rojo = (1, 0, 0) 
        blanco = (1, 1, 1)  # Cambiado a blanco
        alternar_color = True

        for distancia in np.arange(0, longitud + 1, separacion):
            if delta_x == 0:
                x = coordenada_inicio[0]
            else:
                x = coordenada_inicio[0] + distancia * (delta_x / longitud)
            y = coordenada_inicio[1] + pendiente * (x - coordenada_inicio[0])

            # Verificar si las coordenadas están dentro de los límites de la imagen
            if 0 <= x < ancho and 0 <= y < alto:
                # RUIDO EN LAS COORDENADAS:
                x += np.random.normal(loc=0, scale=0.8)
                y += np.random.normal(loc=0, scale=0.8)  # scale= escala el ruido como quiera
                coordenadas.append(((x), (y)))

                # --------- 5. rgb
                for distancia in np.arange(0, longitud + 1, separacion):
                    if delta_x == 0:
                        x = coordenada_inicio[0]
                    else:
                        x = coordenada_inicio[0] + distancia * (delta_x / longitud)
                    y = coordenada_inicio[1] + pendiente * (x - coordenada_inicio[0])

                    # Verificar si las coordenadas están dentro de los límites de la imagen
                    if 0 <= x < ancho and 0 <= y < alto:
                        # RUIDO EN LAS COORDENADAS:
                        x += np.random.normal(loc=0, scale=0.8)
                        y += np.random.normal(loc=0, scale=0.8)  # scale= escala el ruido como quiera
                        coordenadas.append(((x), (y)))

                        # --------- 5.1. Factor de atenuación para el brillo
                        factor_atenuacion = 0.5 - (distancia / longitud)  # Factor que disminuye a medida que avanzas

                        # --------- 5.2. rgb con atenuación
                        if alternar_color:
                            color_rgb = (
                                random.uniform(0.7, 1) * factor_atenuacion,
                                random.uniform(0, 0.3) * factor_atenuacion,
                                random.uniform(0, 0.3) * factor_atenuacion
                            )
                        else:
                            # Cambiado a blanco
                            color_rgb = (
                                1 * factor_atenuacion,
                                1 * factor_atenuacion,
                                1 * factor_atenuacion
                            )

                        colores.append(color_rgb)
                        alternar_color = not alternar_color

        # --------- 6. pintar elipses con ruido gaussiano en los colores
        draw = ImageDraw.Draw(imagen)
        for coord, color in zip(coordenadas, colores):
            x, y = map(int, coord)
            radio = diametro_circulo / 2

            # Verificar si las coordenadas están dentro de los límites de la imagen
            if 0 <= x < ancho and 0 <= y < alto:
                # RUIDO EN EL COLOR
                #scale: aqui añado el ruido
                color_con_ruido = tuple(int((c + np.random.normal(loc=0, scale=0)) * 255 * 0.8) for c in color) #255 * 1, si pongo <1 bajara la intensidad del color en toda la traza
                draw.ellipse([x - radio, y - radio, x + radio, y + radio], fill=color_con_ruido)
                solUNET[y, x] = 255 #pongo blanco la traza en la matriz para UNET

    # Convertir la matriz en una imagen
    imagen_trazas = Image.fromarray(solUNET.astype(np.uint8))
    # --------- 7. Aplicar desenfoque solo a las elipses dibujadas
    puntos_dibujados = imagen.crop((int(coordenada_inicio[0]), int(coordenada_inicio[1]), int(coordenada_fin[0]), int(coordenada_fin[1])))
    puntos_dibujados = puntos_dibujados.filter(ImageFilter.GaussianBlur(radius=0.5))  # radius, para cambiar el ruido
    # Cambiado para aplicar el desenfoque solo a las elipses dibujadas
    imagen.paste(puntos_dibujados, (int(coordenada_inicio[0]), int(coordenada_inicio[1]), int(coordenada_fin[0]), int(coordenada_fin[1])))
    
    # Guardar la imagen
    imagen.save(ruta_guardado_img)
    # Guardar matriz
    imagen_trazas.save(ruta_guardado_unet)



#--------------------------------------------- TIPO 2---------------------------------------------------------------
#traza blanca con el centro ancho y las puntas en punta
    
def trazas_tipo2(ruta_imagen, ruta_guardado_img,ruta_guardado_unet):
    num_trazas =  random.randint(1, 4)
    longitud_minima=40

    cielo = Image.open(ruta_imagen)
    ancho, alto = cielo.size
    
    # SOLUCION UNET
    solUNET = np.zeros((alto, ancho), dtype=np.uint8)  # Crear una matriz de ceros del mismo tamaño que la imagen

    # conf de trazas
    max_grosor = np.random.uniform(2, 6)  
    min_grosor = max_grosor * (2/6)  #para mantener la relacion
    longitud_max = min(ancho, alto) 

    # Crear una nueva imagen con fondo transparente para las trazas
    trazas = Image.new('RGBA', cielo.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(trazas)

    for _ in range (num_trazas):
        # coordenadas de inicio y fin aleatorias
        coordenada_inicio = (np.random.uniform(0, ancho), np.random.uniform(0, alto))
        longitud_linea = np.random.uniform(longitud_minima, longitud_max)  # long random dentro del min y el max
        angulo = np.random.uniform(0, 2*np.pi)
        coordenada_fin = (coordenada_inicio[0] + longitud_linea*np.cos(angulo),
                          coordenada_inicio[1] + longitud_linea*np.sin(angulo))

        #ruido de Perlin para la opacidad
        scale = 10 #+ bajo - ruido
        perlin_values = [noise.pnoise2(x / scale, y / scale) for x, y in zip(range(100), range(100))]

        grosor = np.random.uniform(min_grosor, max_grosor)
        num_pasos = int(longitud_linea) 
        for i in range(num_pasos):

            #posición actual a lo largo de la traza de satélite:
            t = i / num_pasos 
            x = coordenada_inicio[0] + (coordenada_fin[0] - coordenada_inicio[0]) * t
            y = coordenada_inicio[1] + (coordenada_fin[1] - coordenada_inicio[1]) * t

            #opacidad del trozo
            opacidad = (1 - abs(2 * t - 1)) * 150 #150: pero puedo subir hasta 255 para que la opacidad sea maxima en el centro

            #grosor del trozo
            grosor_traza = max_grosor * (1 - abs(2 * t - 1))
            grosor_traza = max(min_grosor, grosor_traza)

            draw.line([(x - grosor_traza / 2, y), (x + grosor_traza / 2, y)],
                      fill=(255, 255, 255, int(opacidad)),
                      width=int(grosor_traza))
            
            # Actualizar la matriz solUNET para marcar los píxeles correspondientes como blancos
            if 0 <= int(x) < ancho and 0 <= int(y) < alto:
                solUNET[int(y), int(x)] = 255

    # Superponer las trazas sobre la imagen de cielo estrellado
    imagen_final = Image.alpha_composite(cielo.convert('RGBA'), trazas)
    
    # Convertir la imagen al modo RGB antes de guardarla como JPEG
    imagen_final = imagen_final.convert("RGB")
    
    # Guardar la imagen final
    imagen_final.save(ruta_guardado_img)

    # Guardar matriz
    imagen_trazas = Image.fromarray(solUNET)
    imagen_trazas.save(ruta_guardado_unet)



#--------------------------------------------- TIPO 3---------------------------------------------------------------
#dos trazas superpuestas, una blanca con poca opacidad y encima una a puntos rojos y blancos   
    
    #ESTE CODIGO HAY QUE LIMPIARLO: con este se guarda toda la traza como una linea continua: (hay q
    #ue limpiar este codigo, no hace falta que haga la funcion esa extra con la linea serviria)
    
def trazas_tipo3(ruta_imagen, ruta_guardado_img, ruta_guardado_unet):
    num_trazas = random.randint(1, 4)
    # Abrir la imagen
    imagen = Image.open(ruta_imagen)

    # Crear una nueva imagen RGBA del mismo tamaño que la original
    nueva_imagen = Image.new("RGBA", imagen.size)

    # Copiar la imagen original a la nueva imagen
    nueva_imagen.paste(imagen, (0, 0))

    # Crear una matriz NumPy del mismo tamaño que la imagen, inicializada con ceros
    solUNET = np.zeros((imagen.height, imagen.width), dtype=np.uint8)

    # Crear un objeto ImageDraw para dibujar sobre la nueva imagen
    dibujo = ImageDraw.Draw(nueva_imagen)
    for _ in range(num_trazas):
        # Generar coordenadas aleatorias para la línea
        x1_linea = random.randint(0, imagen.width)
        y_linea = random.randint(0, imagen.height)

        # Generar una pendiente aleatoria
        pendiente = random.uniform(-1, 1)

        longitud_linea = random.randint(50, 300)
        # Calcular el segundo punto de la línea en función de la pendiente
        x2_linea = x1_linea + longitud_linea  # Longitud de la línea arbitraria
        y2_linea = y_linea + int(longitud_linea * pendiente)

        # Dibujar la línea en la imagen y almacenarla en la matriz solUNET
        dibujo.line([(x1_linea, y_linea), (x2_linea, y2_linea)], fill=(255, 255, 255, 20), width=2)
        trazar_linea_en_matriz(solUNET, x1_linea, y_linea, x2_linea, y2_linea)

        # Anchura de los puntos
        anchura_puntos = 1.8

        # Generar coordenadas para los puntos a lo largo de la línea
        num_puntos = 20
        puntos = []
        for i in range(num_puntos):
            # Calcular las coordenadas x e y del punto, asegurándose de que estén dentro de los límites de la imagen
            x_punto = max(0, min(x1_linea + int((x2_linea - x1_linea) * (i / (num_puntos - 1))), imagen.width - 1))
            y_punto = max(0, min(y_linea + int(pendiente * (x_punto - x1_linea)), imagen.height - 1))
            puntos.append((x_punto, y_punto))
            solUNET[y_punto, x_punto] = 255

        # Alternar entre puntos rojos y blancos
        rojo = True
        for punto in puntos:
            x_punto, y_punto = punto
            correccion = int((anchura_puntos - 1) * 0.5)  # Calcular la corrección para ajustar la anchura
            if rojo:
                dibujo.rectangle([(x_punto - correccion, y_punto - correccion), (x_punto + correccion + 1, y_punto + correccion + 1)], fill=(161, 82, 67, 255))  # Puntos rojos
            else:
                dibujo.rectangle([(x_punto - correccion, y_punto - correccion), (x_punto + correccion + 1, y_punto + correccion + 1)], fill=(134, 127, 109, 255))  # Puntos blancos
            rojo = not rojo  # Alternar entre rojo y blanco en cada iteración

    # Guardar la imagen resultante
    nueva_imagen.save(ruta_guardado_img, format="PNG")

    # Guardar matriz
    imagen_trazas = Image.fromarray(solUNET, mode='L')  # Convertir a imagen de escala de grises
    imagen_trazas.save(ruta_guardado_unet)

def trazar_linea_en_matriz(matriz, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < matriz.shape[0] and 0 <= x0 < matriz.shape[1]:  # Verificar límites
            matriz[y0, x0] = 255
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        if x0 == x1 and y0 == y1:
            if 0 <= y0 < matriz.shape[0] and 0 <= x0 < matriz.shape[1]:  # Verificar límites
                matriz[y0, x0] = 255
            break


#--------------------------------------------- TIPO 4 ---------------------------------------------------------------
#traza blanca normal
def trazas_tipo4(ruta_imagen,ruta_guardado_img,ruta_guardado_unet):
    # Cargar la imagen
    image = Image.open(ruta_imagen)
    num_trazas = random.randint(1, 4)

    # Obtener las dimensiones de la imagen
    width, height = image.size
    # Crear una matriz NumPy del mismo tamaño que la imagen, inicializada con ceros
    solUNET = np.zeros((height, width), dtype=np.uint8)

    # Crear una nueva imagen transparente
    transparent_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Crear un objeto ImageDraw para dibujar en la imagen transparente
    draw = ImageDraw.Draw(transparent_image)

    for _ in range(num_trazas):
        # Coordenadas aleatorias de inicio y fin de la línea
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)

        # Asegurar que las coordenadas estén dentro de los límites de la imagen
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        # Opacidad aleatoria
        alpha = random.randint(20, 255)

        # Dibujar una línea aleatoria en la imagen transparente con opacidad aleatoria
        color = (255, 255, 255, alpha)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
        
        # Obtener los píxeles dibujados en la imagen transparente
        pixels = transparent_image.getdata()

        # Actualizar los valores de la matriz para representar la línea dibujada
        for y in range(height):
            for x in range(width):
                if pixels[y * width + x][3] > 0:  # Verificar si el píxel es transparente
                    solUNET[y, x] = 255  # Blanco

    # Superponer la imagen transparente sobre la imagen original con opacidad
    composite_image = Image.alpha_composite(image.convert('RGBA'), transparent_image)


     # Guardar la imagen resultante
    composite_image.save(ruta_guardado_img, format="PNG")

    # Guardar matriz
    imagen_trazas = Image.fromarray(solUNET)
    imagen_trazas.save(ruta_guardado_unet)

#--------------------------------------------- TIPO 5 ---------------------------------------------------------------
#traza roja normal
def trazas_tipo5(ruta_imagen,ruta_guardado_img,ruta_guardado_unet):
    # Cargar la imagen
    image = Image.open(ruta_imagen)
    num_trazas = random.randint(1, 4)

    # Obtener las dimensiones de la imagen
    width, height = image.size
    # Crear una matriz NumPy del mismo tamaño que la imagen, inicializada con ceros
    solUNET = np.zeros((height, width), dtype=np.uint8)

    # Crear una nueva imagen transparente
    transparent_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Crear un objeto ImageDraw para dibujar en la imagen transparente
    draw = ImageDraw.Draw(transparent_image)

    for _ in range(num_trazas):
        # Coordenadas aleatorias de inicio y fin de la línea
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)

        # Asegurar que las coordenadas estén dentro de los límites de la imagen
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        # Opacidad aleatoria
        alpha = random.randint(20, 255)

        # Dibujar una línea aleatoria en la imagen transparente con opacidad aleatoria
        color = (255, 0, 0, alpha)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
        
        # Obtener los píxeles dibujados en la imagen transparente
        pixels = transparent_image.getdata()

        # Actualizar los valores de la matriz para representar la línea dibujada
        for y in range(height):
            for x in range(width):
                if pixels[y * width + x][3] > 0:  # Verificar si el píxel es transparente
                    solUNET[y, x] = 255  # Blanco

    # Superponer la imagen transparente sobre la imagen original con opacidad
    composite_image = Image.alpha_composite(image.convert('RGBA'), transparent_image)


     # Guardar la imagen resultante
    composite_image.save(ruta_guardado_img, format="PNG")

    # Guardar matriz
    imagen_trazas = Image.fromarray(solUNET)
    imagen_trazas.save(ruta_guardado_unet)