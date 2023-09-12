import pandas as pd
from pathlib import Path
import joblib
import itertools
import numpy as np
import shap 

BASE_DIR = Path(__file__).resolve(strict=True).parent

# On load le modèle 
model = joblib.load(f"{BASE_DIR}/banking_model_20230901135647/model.pkl")
proba_threshold = 0.42

def preparation_file_model(df):
    ###
    #    Fonction permettant de préparer le fichier pour une étude avec le modèle
    ###
    if "DAYS_BIRTH" in df.columns:
        df["ANNEES_AGE"] = (abs(df.DAYS_BIRTH) / 365.25)
        # On élimine l' ancienne variable
        df.drop(["DAYS_BIRTH"], axis = 1, inplace = True)
        
    if "ANNEES_LAST_PHONE_CHANGE" in df.columns:
        df["ANNEES_LAST_PHONE_CHANGE"] = np.round((abs(df.DAYS_LAST_PHONE_CHANGE) / 365.25), 2)
        # On élimine l' ancienne variable
        df.drop(["DAYS_LAST_PHONE_CHANGE"], axis = 1, inplace = True)
    
    # On corrige les erreurs 
    infinity_indices = np.where(np.isinf(df))
    for row_ind, col_ind in zip(*infinity_indices):
        df.iloc[row_ind, col_ind] = df.iloc[:,col_ind].median()
    
    # On récupère les features du modèle
    features = model.named_steps["select_columns"].columns
    
    return df[features]

def application_model(df):
    ###
    #    Fonction permettant d'appliquer le modèle au dataframe
    ###

    # On prédit les probabilité selon le modèle
    # Prédiction d'avoir un prêt
    df["proba_pred_pret"] = np.round(model.predict_proba(df)[:, 0], 2)
    # Prédiction de ne pas avoir un prêt
    df["proba_pred_non_pret"] = np.round(1 - df["proba_pred_pret"], 2)
    
    # Résultat selon le threshold du modèle établit. 
    # Si au dessus, la valeur = 1, ce qui correspond à la non obtention d'un prêt
    df["prediction"] = np.where(df["proba_pred_non_pret"] >= proba_threshold, 1, 0)
    
    df["prediction_pret"] = np.where(df["prediction"] == 1, "Non pret", "Pret")
    df = scoring_pret(df)

    return df

def feature_importance_client(df): 
    ###
    #    Fonction permettant de mesurer l'importance des features pour le client
    ###
    
    features = model.named_steps["select_columns"].columns
    
    x_train_preprocessed = model[:-1].transform(df[features])
    
    selected_cols = df[features].columns[model.named_steps["feature_selection"].get_support()]
    
    x_train_preprocessed = pd.DataFrame(x_train_preprocessed, columns = selected_cols)
    
    # on charge l'objet explainer du modèle
    with open(f"{BASE_DIR}/explainer_model.pkl", 'rb') as explainer_file:
        explainer_model = joblib.load(explainer_file)

    shap_values = explainer_model(x_train_preprocessed)
    
    df_sk_shape = pd.DataFrame({'SK_ID_CURR': df["SK_ID_CURR"].values, 'value_total': shap_values.values.sum(axis=1)})

    df_feat_shape = pd.DataFrame(shap_values.values, columns = selected_cols)

    df_shape_score = pd.concat([df_sk_shape, df_feat_shape], axis = 1)
    
    return df_shape_score


def scoring_pret(df):
    
    # On calcul un score selon si le client a obtenu un prêt ou pas
    # Ce score donne une autre appréciation des probabilités, plus parlant pour un consommateur
    min_value = 1 - proba_threshold  # Minimum value of proba_pred_pret
    max_value = 1 - proba_threshold
    
    df.loc[df.prediction_pret == "Pret", "score"] = (df["proba_pred_pret"] - min_value) / (1 - min_value)
    df.loc[df.prediction_pret == "Non pret", "score"] = (1 - (df["proba_pred_pret"]) / (max_value)) * - 1
    df.loc[:, "score"] = np.round(df.loc[:, "score"], 4)
    
    lettres = ['a', 'b', 'c', 'd', 'e', 'f']
    signes = ['++', '+', '-', '--']

    # On génére 2 dictionnaires
    bon_clients = {}
    mauvais_clients = {}

    point_decr = (100 / ((len(lettres) * len(signes)) / 2)) / 100

    point = 1
    for l, s in itertools.product(lettres[:3], signes):
        key = l + s
        bon_clients[key] = np.round(point, 2)
        point -= point_decr
    
    point = 0
    for l, s in itertools.product(lettres[3:], signes):
        key = l + s
        mauvais_clients[key] = np.round(point, 2)*-1
        point += point_decr

    condition1 = df.prediction_pret == "Pret"
    condition2 = df.prediction_pret == "Non pret"

    for keys, items in bon_clients.items():
        df.loc[(condition1) & (df.score <= items), "score_note"] = keys

    for keys, items in mauvais_clients.items():
        df.loc[(condition2) & (df.score <= items), "score_note"] = keys


    df.loc[:, "score"] = np.round(df.loc[:, "score"]*100, 2)
    
    return df


