import pandas as pd
from pathlib import Path
import numpy as np
import shap 
from mlflow.sklearn import load_model

BASE_DIR = Path(__file__).resolve(strict=True).parent

# On load le modèle 
#model = joblib.load(f"{BASE_DIR}/banking_model_20230915203320.pkl")
model = load_model("banking_model_20230915203320")
proba_threshold = 0.42

def application_model(df):
    ###
    #    Fonction permettant d'appliquer le modèle au dataframe
    ###

    # On prédit les probabilité selon le modèle
    # Prédiction d'avoir un prêt
    df["proba_pred_pret"] = np.round(model.predict_proba(df)[:, 0], 2)
    # Prédiction de ne pas avoir un prêt
    df["proba_pred_non_pret"] = np.round(1 - df["proba_pred_pret"], 2)
    
    return df
