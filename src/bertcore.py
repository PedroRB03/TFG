
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score,accuracy_score
from scipy.special import softmax
import torch
import shutil
import os
from pathlib import Path
import torch.nn.functional as F
from transformers import set_seed
from optuna import TrialPruned
from common import should_prune, get_class_weights, printtest
import time

# Función para calcular métricas.
def compute_metrics(FUZZY, eval_pred):
    logits, labels = eval_pred

    if FUZZY:
        probs = logits.squeeze()
        predictions= np.round(probs).astype(int) # Redondeamos probabilidades.
        labels = np.round(labels).astype(int)
        predictions = np.clip(predictions, 0, 1) # Aseguramos el rango [0,1].
        roc_auc = roc_auc_score(labels, probs) 

    else:
        probs = softmax(logits, axis=-1)
        predictions = np.argmax(logits, axis=-1)
        roc_auc = roc_auc_score(labels, probs[:, 1]) 
        
    return {
        'macro_f1': f1_score(labels, predictions, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'matthews_corrcoef': matthews_corrcoef(labels, predictions),
        'roc_auc_ovr': roc_auc,
        'accuracy': accuracy_score(labels, predictions),
    }

# Permite entrenar el modelo con los parámetros dados y muestra una media de las métricas de evaluación.
def testbert(study,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL,BALANCED_CW=False,FUZZY_BAL_CRISP=True,model_name="microsoft/deberta-v3-small",NO_TRAIN=False,tmodel_name="microsoft/deberta-v3-small"):
    
    param_grid = study.best_params # Cargamos mejores hiperparámetros.

    results, times = trainbert(param_grid,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL,BALANCED_CW,FUZZY_BAL_CRISP,model_name,NO_TRAIN=NO_TRAIN,tmodel_name=tmodel_name)

    metric_names = ['eval_macro_f1', 'eval_balanced_accuracy', 'eval_matthews_corrcoef', 'eval_roc_auc_ovr','eval_accuracy']

    
    printtest(SEEDS,results,times,metric_names)

# Carga y aplica la función de tokenización a los datasets de entrenamiento, validación y test y los devuelve.
def obtain_tokenized(tokenize_function, u_file, seed):
    train_ds = pd.read_pickle(u_file+str(seed)+"train.pkl")
    eval_ds = pd.read_pickle(u_file+str(seed)+"eval.pkl")
    test_ds = pd.read_pickle(u_file+str(seed)+"test.pkl")

    # Eliminamos columnas con texto pasado por TF-IDF.
    train_ds.drop(columns=["vec"], inplace=True)
    eval_ds.drop(columns=["vec"], inplace=True)
    test_ds.drop(columns=["vec"], inplace=True)

    train_ds = Dataset.from_pandas(train_ds)
    eval_ds = Dataset.from_pandas(eval_ds)
    test_ds = Dataset.from_pandas(test_ds)
    
    t_train = train_ds.map(tokenize_function, batched=True)
    t_eval = eval_ds.map(tokenize_function, batched=True)
    t_test = test_ds.map(tokenize_function, batched=True)

    return t_train, t_eval, t_test

# Función para sobrescribir el método compute_loss y aplicar pesos de clase
def weighted_compute_loss(class_weights, outputs, labels, num_items_in_batch=None):
    logits = outputs.logits

    # Fuzzy (num_labels == 1).
    if logits.shape[-1] == 1:

        label_idx = torch.bucketize( # Calculamos a qué peso de clase le corresponde cada ejemplar.
            labels,
            boundaries=torch.tensor([0.165, 0.495, 0.83], device=labels.device) # [(0+0.33)/2,(0.33+0.66)/2,(0.66+1)/2].
            # Usamos intervalos para asignar pesos de clase ya que tenemos valores continuos.
        )
        sample_weights = class_weights.to(labels.device)[label_idx]
        
        loss = F.mse_loss(logits.squeeze(), labels.squeeze(),reduction="none")
        loss = (loss * sample_weights).mean()

    # Crisp 
    else:
        loss = F.cross_entropy(logits, labels,weight=class_weights.to(logits.device))

    return loss

# Devuelve la clase Tokenizer correspondiente al modelo y una función para obtener tokens.
def get_tokenizer(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                use_fast=True,
                                                extra_special_tokens=['[URL]','[USER]']
                                                )
    
    def tokenize_function(examples):
        return tokenizer(examples["txt"], padding="max_length", truncation=True, max_length=256)

    return tokenizer, tokenize_function

# Entrena el modelo con las semillas y parámetros dados y devuelve las métricas de evaluación.
def trainbert(param_grid,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL=False,BALANCED_CW=False,FUZZY_BAL_CRISP=True,MODEL_NAME = "microsoft/deberta-v3-small",trial=None,NO_TRAIN=False,tmodel_name="microsoft/deberta-v3-small"):

    tokenizer, tokenize_function = get_tokenizer(tmodel_name)

    results = [] # Resultados de cada semilla.

    fit_t = []
    model_name = MODEL_NAME

    for step,seed in enumerate(SEEDS):
        if trial is not None:
            if trial.should_prune():
                raise TrialPruned(f"Podada semilla {seed} por decisión del pruner.")
            if should_prune(trial,results,len(SEEDS),eval_name='eval_macro_f1'):
                raise TrialPruned(f"Podada semilla {seed} por media optimista inferior al mejor estudio.")
            

        print(f"\n--- Entrenando Seed {seed} ---")

        set_seed(seed) 

        t_train, t_eval, t_test = obtain_tokenized(tokenize_function,u_file,seed) # Datasets tokenizados.

        dname = RESULTS+"/seed"+str(seed)
        Path(dname).mkdir(parents=True, exist_ok=True) # Creamos ruta si no existe.

        shutil.rmtree(dname) # Borramos directorio y archivos donde se guardan los checkpoints.
        os.mkdir(dname) # Lo volvemos a crear.

        if NO_TRAIN:
            model_name = MODEL_NAME+"/bert_model"+str(seed)

        if FUZZY:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,dtype=torch.float32) # Para regresión
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,dtype=torch.float32)
        model.resize_token_embeddings(len(tokenizer))
        
        # Hiperparámetros.
        training_args = TrainingArguments(
            output_dir=dname,
            eval_strategy='epoch',
            save_strategy='best',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=param_grid['gradient_accumulation_steps'],
            learning_rate=param_grid['learning_rate'],
            num_train_epochs=param_grid['num_train_epochs'],
            warmup_steps=param_grid['warmup_steps'],
            weight_decay=param_grid['weight_decay'],
            logging_steps=100,
            seed=seed,
            data_seed=seed,
            metric_for_best_model='macro_f1',
            load_best_model_at_end=True,
            #full_determinism=True, # viene bien para determinadas pruebas pero hace que el entrenamiento sea MUY lento
        )
        
        callbacks = []

        if EARLY_STOP > 0: # Si EARLY_STOP no es cero, incluimos el callback.
            callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP)]

        if BALANCED_CW: # Calculamos pesos de clase.
            class_weights = get_class_weights(t_train["label"], FUZZY_BAL_CRISP) 
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            
        def _c_loss(outputs, labels, num_items_in_batch=None):
            return weighted_compute_loss(class_weights,outputs,labels,num_items_in_batch)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=t_train,
            eval_dataset=t_eval,
            compute_metrics=lambda eval_pred: compute_metrics(FUZZY, eval_pred),
            callbacks=callbacks,
            compute_loss_func=None if not BALANCED_CW else _c_loss
        )

        st = time.time()
        if not NO_TRAIN:
            trainer.train()
        fit_t.append(time.time()-st)

        if SAVE_MODEL and not NO_TRAIN: # Guardamos modelo entrenado
            sdname = RESULTS+"/bert_model"+str(seed)
            Path(sdname).mkdir(parents=True, exist_ok=True)
            trainer.save_model(sdname)
            print(f"Modelo guardado en semilla {seed}.")
        

        if USE_TEST: # Calcula métricas con datos de test o validación.
            eval_stats = trainer.evaluate(eval_dataset=t_test)
        else:
            eval_stats = trainer.evaluate(eval_dataset=t_eval)
        if trial is not None:
            trial.report(eval_stats['eval_macro_f1'],step=step)
        results.append(eval_stats)

    return results, fit_t

