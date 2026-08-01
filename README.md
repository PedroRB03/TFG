
# TFG - Detección de ragebait/trolling

En este repositorio se encuentran los ficheros relacionados con el TFG
de detección de ragebait/trolling.

El proyecto sigue la siguiente estructura:
- data: contiene archivos relacionados con el dataset.
    - pkls: carpeta que contiene archivos ".pkl".
        - crisp: contiene datos de entrenamiento, evaluación y test CRISP para cada semilla.
        - fuzzy: contiene datos de entrenamiento, evaluación y test FUZZY para cada semilla.
    - trolling.xlsx: dataset original.
- results: contiene modelos entrenados y otros archivos generados.
- src: contiene los scripts de Python para el uso y entrenamiento de los distintos modelos.
    - datagen.py: permite generar los archivos de datos preprocesados para el uso de los modelos. Adicionalmente, guarda el modelo TF-IDF entrenado.
    - common.py: contiene funciones de uso general además de entrenamiento para modelos LGBM y SVM.
    - lgbm.py: script para el uso, guardado o búsqueda de hiperparámetros del modelo LGBM.
    - svm.py: script para el uso, guardado o búsqueda de hiperparámetros del modelo SVM.
    - bertcore.py: contiene funciones para el entrenamiento, test y guardado de modelos mDeBERTaV3.
    - bert.py: script para el uso, guardado o búsqueda de hiperparámetros del modelo mDeBERTaV3.
- requirements.txt : Requisitos de paquetes para python. Es posible que hayan más paquetes de los necesarios, si lo prefiere, instale solo los paquetes que necesite cada archivo. 
- params.ini: Archivo de configuración de las distintas scripts.

La carpeta "old" se puede ignorar y todo commit previo a su creación.
