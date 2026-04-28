import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/Users/pedro/Desktop/TFG/tesseract/tesseract.exe' # PATH


def leer_img(img)->str:

    h,w = img.shape # Altura y anchura

    if h < 256 or w < 256: # Aumentamos escala si la imagen es muy pequena
        img = cv2.resize(img,[w*2,h*2],interpolation=cv2.INTER_LANCZOS4)

    tresh,_ = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Obtenemos umbral

    img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_CONSTANT,value=tresh) # Metemos borde por si hay letras pegadas al borde

    #cv2.imshow("prueba",img)
    #cv2.waitKey(0)

    all_char = [ # Vamos a intentar filtrar ruido en el texto
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z',
        'á', 'é', 'í', 'ó', 'ú', 'ü',

        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'Á', 'É', 'Í', 'Ó', 'Ú', 'Ü',

        '.',',',':',';',' ','\t','\n','?','!'

    ]

    charset = set(all_char) # Un set es mas rapido

    text = pytesseract.image_to_string(img ) # Leemos texto

    text = ''.join(c for c in text if c in charset) # Filtramos ruido

    return text


#img = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)
#print(leer_img(img))