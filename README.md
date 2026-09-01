
# TFG - Detección automatizada de ragebait en contenidos de redes sociales

En este repositorio se encuentran los ficheros con código fuente y ejemplos entrenados utilizados en el TFG adjunto. Trabajo realizado
por el alumno Pedro Luis Roldán Benítez para la Universidad de Cádiz con la Tutoría de Gabriel Guerrero-Contreras.

El proyecto sigue la siguiente estructura:
- data: contiene archivos relacionados con el dataset.
    - pkls: carpeta que contiene archivos ".pkl".
        - crisp: contiene datos generados de entrenamiento, evaluación y test CRISP para cada semilla.
        - fuzzy: contiene datos generados de entrenamiento, evaluación y test FUZZY para cada semilla.
    - trolling.xlsx: dataset original.
- study: contiene estudios realizados de Optuna para cada configuración de los modelos.
- results: contiene modelos entrenados y otros archivos generados.
- src: contiene los scripts de Python para el uso y entrenamiento de los distintos modelos.
    - datagen.py: permite generar los archivos de datos preprocesados para el uso de los modelos. Adicionalmente, guarda el modelo TF-IDF entrenado.
    - datastudy.py: genera gráficas y tablas sobre el la fuente de datos.
    - common.py: contiene funciones de uso general además de entrenamiento para modelos LGBM y SVM.
    - lgbm.py: script para la evaluación, entrenamiento, guardado o búsqueda de hiperparámetros del modelo LGBM.
    - svm.py: script para la evaluación, entrenamiento, guardado o búsqueda de hiperparámetros del modelo SVM.
    - bertcore.py: contiene funciones para el entrenamiento, test y guardado de modelos mDeBERTaV3.
    - bert.py: script para la evaluación, entrenamiento, guardado o búsqueda de hiperparámetros del modelo mDeBERTaV3.
    - main.py: permite cargar modelos y hacer predicciones sobre un conjunto de datos pasado.
    - optstudy.py: muestra una gráfica con la evolución de un estudio de Optuna además de algunos datos de interés como número de pruebas o pruebas podadas.
- requirements.txt : Requisitos de paquetes para python. Es posible que hayan más paquetes de los necesarios, si lo prefiere, instale solo los paquetes que necesite cada archivo. 
- params.ini: Archivo de configuración de las distintas scripts.

La carpeta "old" se puede ignorar y todo commit previo a su creación.

