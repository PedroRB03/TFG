
# TFG - Detección de ragebait/trolling

En este proyecto se encuentran los ficheros relacionados con el TFG
de detección de ragebait/trolling.

Cabe destacar los siguientes archivos de interés:
- data/trolling.xlsx : Es el dataset de trolling. 
- src/datagen.py : Toma el archivo 'trolling.xlsx' y genera archivos de entrenamiento, evaluación y test por cada semilla configurada. Genera dos versiones de archivos por semilla, uno para lógica crisp (es decir, con etiqueta a 0 o 1) y otra para lógica fuzzy (etiquetas en el intervalo \[0,1\]).
- src/lgbm.py : Permite evaluar el modelo o buscar hiperparámetros del modelo LGBM con los datos de 'rb_db' generados por datagen.py.
- src/svm.py : Permite evaluar el modelo o buscar hiperparámetros del modelo SVM con los datos de 'rb_db' generados por datagen.py.
- src/bert.py : Permite evaluar el modelo Microsoft/DeBERTaV3-base con los datos de 'rb_db' o 'rbf_db' (si es fuzzy) generados por datagen.py.
- requirements.txt : Requisitos de paquetes para python. Es posible que hayan más paquetes de los necesarios, si lo prefiere, instale solo los paquetes que necesite cada archivo. 

Los archivos de Python suelen tener la declaración de algunos parámetros al principio de estos. Dichos parámetros suelen controlar las seeds o las rutas de los archivos requeridos.

En la carpeta "data" se encuentran los datasets usados en los modelos.
Por otro lado en "results" se encuentran los resultados del entrenamiento de los respectivos modelos.

La carpeta Old se puede ignorar y todo commit previo a su creación.
