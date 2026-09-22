import sys
import pandas as pd
from common import get_params

# Comprueba si dos conjuntos comparten elementos.
# Si no comparten elementos, la suma de sus cardinales es el cardinal de su unión. 
# En caso de no compartir elementos devuelve True.
def _check_common(set1,set2):
    return len(set1)+len(set2) == len(set1.union(set2))

if __name__ == "__main__":
            
    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")

    SEEDS = params["COMMON"]["SEEDS"]
    TEST_PERCENT = params["DATAGEN"]["TEST_PERCENT"]
    EVAL_PERCENT = params["DATAGEN"]["EVAL_PERCENT"]
    FILE = params["COMMON"]["FILE"]
    BALANCING = params["DATAGEN"]["BALANCING"]
    ##

    print("Validando conjuntos por semilla.")
    for seed in SEEDS:
        print(f"SEED {seed}:")
        trn = pd.read_pickle(FILE+"/s"+str(seed)+"train.pkl")
        ev = pd.read_pickle(FILE+"/s"+str(seed)+"eval.pkl")
        tst = pd.read_pickle(FILE+"/s"+str(seed)+"test.pkl")

        s_trn = set(trn['txt'].astype(str))
        s_ev = set(ev['txt'].astype(str))
        s_tst = set(tst['txt'].astype(str))

        trn_dupe = trn.duplicated(subset=["txt"],keep=False)
        ev_dupe = ev.duplicated(subset=["txt"],keep=False)
        tst_dupe = tst.duplicated(subset=["txt"],keep=False)

        problems = 0

        print("Buscando problemas...")

        if len(trn_dupe) != len(s_trn):
            problems +=1
            print("- Hay elementos duplicados en Entrenamiento Fuzzy.")
        if len(trn_dupe) != len(s_trn):
            problems +=1
            print("- Hay elementos duplicados en Entrenamiento Crisp.")
        if len(ev_dupe) != len(s_ev):
            problems +=1
            print("- Hay elementos duplicados en Validación.")
        if len(tst_dupe) != len(s_tst):
            problems +=1
            print("- Hay elementos duplicados en Test.")
        
        if not _check_common(s_tst,s_trn): 
            problems +=1
            print("- Test y Entrenamiento Fuzzy comparten muestras.")
        if not _check_common(s_tst,s_trn): 
            problems +=1
            print("- Test y Entrenamiento Crisp comparten muestras.")
        if not _check_common(s_ev,s_trn): 
            problems +=1
            print("- Validación y Entrenamiento Fuzzy comparten muestras.")
        if not _check_common(s_ev,s_trn): 
            problems +=1
            print("- Validación y Entrenamiento Crisp comparten muestras.")

        if not _check_common(s_ev,s_tst):
            problems +=1
            print("- Validación y Test comparten muestras.")

        if problems == 0:
            print("No se han encontrado problemas.")
        else:
            print(f"Se han encontrado {problems} problemas.")

        print(f"Entrenamiento contiene {len(trn)} filas ({len(trn[trn["label"] == 0])} Normal y {len(trn[trn["label"] == 1])} Trolling).")
        print(f"Validación contiene {len(ev)} filas ({len(ev[ev["label"] == 0])} Normal y {len(ev[ev["label"] == 1])} Trolling).")
        print(f"Test contiene {len(tst)} filas ({len(tst[tst["label"] == 0])} Normal y {len(tst[tst["label"] == 1])} Trolling).")
        