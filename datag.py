import albumentations as A
import cv2
import os
import datetime
import time

global TRANSFORM
global NUM_TO_GENERATE
global WRITE
global VERBOSE
global INTERPOLACION
global RESIZE_TYPE
global MANTENER_ASPECTO

MANTENER_ASPECTO = True
RESIZE_TYPE = 'min'
INTERPOLACION = cv2.INTER_NEAREST
VERBOSE = True
NUM_TO_GENERATE = 10
WRITE = False

TRANSFORM = A.Compose([
        # A.RandomRotate90(),
        A.HorizontalFlip(),
        # A.Transpose(),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.11, rotate_limit=20, p=0.4),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])


def guardar_img(img_array, path_out, num=0):
    """
    img_array = img leida por cv2\n
    path_out = dir de salida\n
    num = valor para numerar la img\n
    guarda la imagen en el path indicado,
    el nombre esta dado por el valor (en caso de pasarlo como parametro),
    y la fecha actual del sistema junto con el tiempo\n
    salida en formato jpg
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        datos_img = '{}/{}_{}.jpg'.format(path_out, num, 'aumentado')
        cv2.imwrite(datos_img, img_array)
    except Exception as e:
        print(f'error en guardar_img: {e}')
    return


# hace el aumento de una sola imagen
# img_array = img leida por opencv
def aumentar_imagen(img_array, path_out, img_count=0, write=WRITE):
    new_img = TRANSFORM(image=img_array)['image']
    if write:
        guardar_img(new_img, path_out, img_count)
    return new_img


# dada una ruta se extrae los nombres de todos los archivos como un vector
def extraer_nombres(path_in, negative=None):
    path_in = os.path.normpath(path_in)
    if negative is None:
        negative = []
    nombres = []
    for img_name in os.listdir(path_in):
        if img_name in negative:
            continue
        nombres.append(img_name)
    return nombres
