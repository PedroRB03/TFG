
# TFG - Detección de ragebait/trolling

En este proyecto se encuentran los ficheros relacionados con el TFG
de detección de ragebait/trolling.

Cabe destacar los siguientes archivos de interés:
- trolling.xlsx : Es el dataset de trolling. 
- filtrado.py : Toma el archivo 'trolling.xlsx', normaliza el texto y genera un archivo de pandas con etiqueta 0 o 1 en cada fila dependiendo de si los tres jueces de 'trolling.xlsx' coincidían si era trolling o coincidían si NO era trolling.
- prueba_lgbm.py : Permite evaluar con KCrossValidation o buscar hiperparámetros del modelo LGBM con los datos de 'rb_db.pkl' generados por filtrado.py.
- prueba_svm.py : Permite evaluar con KCrossValidation o buscar hiperparámetros del modelo SVM con los datos de 'rb_db.pkl' generados por filtrado.py.
- prueba_bert.py : Permite evaluar con KCrossValidation el modelo Microsoft/DeBERTaV3-base con los datos de 'rb_db.pkl' generados por filtrado.py.
- requirements.txt : Requisitos de paquetes para python. Es posible que hayan más paquetes de los necesarios, si lo prefiere, instale solo los paquetes que necesite cada archivo. 

Los archivos de python suelen tener la declaración de algunos parámetros al principio de estos. Dichos parámetros suelen controlar la seed o las rutas de los archivos requeridos.

La carpeta Old se puede ignorar y todo commit previo a su creación.
