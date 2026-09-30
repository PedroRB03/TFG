
# TFG - Detección de ragebait/trolling

En este repositorio se encuentran los ficheros con código fuente y ejemplos entrenados utilizados en el TFG adjunto. Trabajo realizado
por el alumno Pedro Luis Roldán Benítez para la Universidad de Cádiz con la Tutoría de Gabriel Guerrero-Contreras.

## Requisitos

* Python **3.13.13**
* Dependencias indicadas en `requirements.txt`
* Carpeta `best/` con los modelos preentrenados proporcionados junto con el proyecto.

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Estructura

```text
.
├── src/                # Código fuente y scripts de Python
├── data/               # Datos originales y conjuntos preprocesados
├── study/              # Estudios y bases de datos SQLite
├── best/               # Modelos preentrenados
├── params.ini          # Configuración del proyecto
└── requirements.txt    # Dependencias de Python
```

Los principales scripts de `src/` son:

* `datagen.py`: generación y preprocesado de los conjuntos de datos.
* `datastudy.py`: análisis del conjunto de datos.
* `svm.py`: entrenamiento y evaluación de SVM.
* `lgbm.py`: entrenamiento y evaluación de LightGBM.
* `bert.py`: entrenamiento y evaluación de DeBERTaV3.
* `main.py`: inferencia utilizando los modelos entrenados.
* `testperform.py`: medición de tiempos de inferencia.
* `testdata.py`: comprobación de integridad de los conjuntos de datos.
* `optstudy.py`: consulta y análisis de estudios de Optuna.

## Ejecución

Los scripts se ejecutan desde la raíz del repositorio. Por defecto utilizan `params.ini`, aunque se puede proporcionar otro archivo de configuración como argumento.

### Generar los datos

Para LGBM:
```bash
python src/datagen.py best/lgbm-smote-1-1.ini
```
Para SVM:
```bash
python src/datagen.py best/svm-poly-enn-1-2.ini
```
Para DeBERTaV3:
```bash
python src/datagen.py best/bert-fuzzy-cw.ini
```

### Evaluar modelos

Para LGBM:
```bash
python src/lgbm.py best/lgbm-smote-1-1.ini
```
Para SVM:
```bash
python src/svm.py best/svm-poly-enn-1-2.ini
```
Para DeBERTaV3:
```bash
python src/bert.py best/bert-fuzzy-cw.ini
```

### Realizar predicciones

`main.py` acepta archivos `.csv` o `.pkl`. Se utiliza la primera columna como conjunto de textos.

```bash
python src/main.py prueba.csv -c params.ini -m all
```

Opciones principales:

* `-m, --model`: modelo a utilizar (`svm`, `lgbm`, `bert` o `all`).
* `-c, --config`: archivo de configuración.
* `-p, --proba`: muestra las salidas/probabilidades de los modelos.
* `-t, --time`: muestra el tiempo de inferencia.
* `-l, --limit`: limita el número de muestras procesadas.

Por ejemplo:

```bash
python src/main.py prueba.csv -c params.ini -m all -t -p -l 3
```

Para utilizar los modelos finales, la carpeta `best/` debe estar situada en la raíz del proyecto. El `params.ini` incluido contiene la configuración necesaria para utilizar los modelos finales de la semilla 600.

### Comprobaciones y análisis

Después de utilizar datagen.py:
```bash
python src/testdata.py params.ini
python src/testperform.py params.ini
python src/datastudy.py params.ini
```

Para consultar un estudio de Optuna (para lgbm, por ejemplo):

```bash
python src/optstudy.py study/lgbm-study-smote-1-1.db lgbm-study
```

## Configuración

El archivo `params.ini` contiene las secciones `COMMON`, `DATAGEN`, `LGBM`, `SVM` y `BERT`. Permite configurar las semillas, rutas de datos y modelos, parámetros de preprocesado, balanceo y opciones de entrenamiento y optimización.

La documentación completa de estos parámetros y de los experimentos realizados se encuentra en la memoria del TFG.
