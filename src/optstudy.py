import sys
import optuna
import optuna.visualization.matplotlib as optuna_plt
from datetime import timedelta
import matplotlib.pyplot as plt
from warnings import simplefilter

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Se deben especificar ruta y nombre del estudio en este orden.")
    else:
        simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
     
        study = optuna.study.load_study(storage=sys.argv[1],study_name=sys.argv[2])

        ctrials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print(f"=== RESUMEN DEL ESTUDIO ===")

        df = study.trials_dataframe()
        df['sec'] = df['duration'].dt.total_seconds()
        c_dur, p_dur = df[df.state == 'COMPLETE']['sec'], df[df.state == 'PRUNED']['sec']

        print(f"Trials: {len(study.trials)} | Pruned: {len(p_dur)} | Mejor F1: {study.best_value:.4f}")
        print(f"Duración media Complete: {timedelta(seconds=round(c_dur.mean() or 0))} | Pruned: {timedelta(seconds=round(p_dur.mean() or 0))} | Tiempo Total: {timedelta(seconds=round(df['sec'].sum()))}")
        print(f"Mejores parámetros: {study.best_params}")

        optuna_plt.plot_optimization_history(study)
        plt.title("Evolución del F1 Score")
        plt.show()