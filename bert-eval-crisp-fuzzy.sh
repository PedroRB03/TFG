#!/bin/bash

echo "BÚSQUEDA FUZZY"
python src/datagen.py params/fuzzy.ini
python src/bert.py params/fuzzy.ini

echo "BÚSQUEDA CRISP"
python src/datagen.py params/crisp.ini
python src/bert.py params/crisp.ini
