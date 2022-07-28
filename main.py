import numpy as np
import cv2
import os
import pytesseract
import matplotlib.pyplot as plt

# Seta path do tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# files folder
fileFolder = "images/"

# get files
files = os.listdir(fileFolder)

#create output file
outputFile = 'output.txt'
try:
    os.remove(outputFile)
except OSError as e:
    print(e)

outputFile = open(outputFile, 'a')

# TO-DO Looping no diretório pegando arquivos de imagem
for filename in files:
    print(filename)
    # Leitura da imagem
    img = cv2.imread(fileFolder + filename)
    cv2.imshow('original', img)
    cv2.waitKey(0)

    # Blur bilateral para desfocar as bordas necessário para que o algoritmo do canny tem mais performance para localizar contornos relevantes
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow('Blur bilateral', bilateral)
    cv2.waitKey(0)

    # Transforma a imagem para o tome preto e branco (cinza) para diminuir o ruido das cores
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grey tones', gray)
    cv2.waitKey(0)

    # Pegando contornos da imagem com o canny, gera imagem com fundo preto e com todas os contornos
    edged = cv2.Canny(gray, 30, 200)
    cv2.imshow('Contornos', edged)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print('Numero de contornos encontrados: ', len(contours))

    # Looping em cada contorno para encontrar os padrões de um retangulo
    largest_rectangle = [0, 0]
    for cnt in contours:
        # Realiza uma aproximação dos contornos disponiveis, tentando gerar o mais próximo de um poligono retangulo
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        # Procurando um contorno de quatro lados no poligono gerado visando que as placas são retangulos, procuramos um poligono de 4 faces.
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > largest_rectangle[0]:
                # Procurando contorno de maior area.
                largest_rectangle = [cv2.contourArea(cnt), cnt, approx]

    # Parametros para desenhar um retangulo ao redor da placa
    x, y, w, h = cv2.boundingRect(largest_rectangle[2])

    # Crop da placa.
    roi = img[y:y+h, x:x+w]

    # Plotando o roi gerado da placa
    plt.imshow(roi, cmap='gray')
    plt.show()



    # Captura dos caracteres da placa, dentro de uma lista de possiveis characteres, sendo de a~z e 0~9, sem characteres especiais
    placa = pytesseract.image_to_string(roi, config='--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    print("Placa aproximada do carro: ", placa)

    if len(placa) == 0:
        outputFile.write("Leitura incorreta para o arquivo: " + filename + '\n')
    else:
        outputFile.write("O numero da placa é: " + placa.replace('\n', '') + '\n')

cv2.destroyAllWindows()
print("Fim do processamewnto")