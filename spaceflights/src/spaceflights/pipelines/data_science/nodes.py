import logging
import importlib
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

def get_class(class_path: str):
    """Helper para importar uma classe a partir de uma string (ex: sklearn.ensemble.RandomForestClassifier)."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_options: Dict[str, Any],
) -> Any:
    """Treina um modelo de classificação com base nas opções fornecidas."""
    logger = logging.getLogger(__name__)

    model_type = model_options["type"]
    model_params = model_options.get("params", {})
    use_smote = model_options.get("use_smote", True)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    if use_smote:
        logger.info("Aplicando SMOTE para balanceamento de classes.")
        sm = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    else:
        X_tr_res, y_tr_res = X_tr, y_tr

    model_class = get_class(model_type)
    model = model_class(**model_params)

    logger.info(f"Treinando modelo: {model_type}")

    if "catboost" in model_type.lower():
        cat_features = model_options.get("cat_features")
        if cat_features is None:
            cat_features = list(X_tr_res.select_dtypes(include=['category', 'object']).columns)
            logger.info(f"CatBoost: Features categóricas detectadas automaticamente: {cat_features}")
        
        for col in cat_features:
            X_tr_res[col] = X_tr_res[col].astype('category')
            X_val[col] = X_val[col].astype('category')

        model.fit(X_tr_res, y_tr_res, cat_features=cat_features)
    else:
        model.fit(X_tr_res, y_tr_res)

    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, zero_division=0)
    logger.info(f"Métricas de validação para {model_type} -> Acurácia: {acc:.3f} | F1-Score: {f1:.3f}")

    return model

def tune_and_train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_options: Dict[str, Any],
    tuning_options: Dict[str, Any],
) -> Any:
    """Otimiza hiperparâmetros e treina o modelo final."""
    logger = logging.getLogger(__name__)

    model_type = model_options["type"]
    param_grid = tuning_options["grids"].get(model_type, {})

    if not param_grid:
        logger.warning(f"Nenhuma grade de hiperparâmetros encontrada para {model_type}. Pulando otimização.")
        return train_model(X_train, y_train, model_options)

    logger.info(f"Iniciando GridSearchCV para {model_type}")

    model_class = get_class(model_type)
    estimator = model_class(random_state=42)

    # Lida com SMOTE dentro do pipeline de CV para evitar data leakage
    if tuning_options.get("use_smote", True):
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', estimator)
        ])
        # Ajusta os nomes dos parâmetros para o pipeline
        param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
    else:
        pipeline = estimator

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    logger.info(f"Melhores parâmetros para {model_type}: {grid_search.best_params_}")
    logger.info(f"Melhor F1-Score (CV): {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_


def get_class(class_path: str):
    """Helper para importar uma classe a partir de uma string (ex: sklearn.ensemble.RandomForestClassifier)."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_options: Dict[str, Any],
) -> Any:
    """Treina um modelo de classificação com base nas opções fornecidas."""
    logger = logging.getLogger(__name__)

    # Parâmetros do modelo e SMOTE
    model_type = model_options["type"]
    model_params = model_options.get("params", {})
    use_smote = model_options.get("use_smote", True)

    # Divisão treino/validação para avaliação interna rápida
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Balanceamento de classes com SMOTE, se ativado
    if use_smote:
        logger.info("Aplicando SMOTE para balanceamento de classes.")
        sm = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    else:
        X_tr_res, y_tr_res = X_tr, y_tr

    # Instancia o modelo a partir da string
    model_class = get_class(model_type)
    model = model_class(**model_params)

    logger.info(f"Treinando modelo: {model_type}")

    # Tratamento especial para CatBoost com features categóricas
    if "catboost" in model_type.lower():
        cat_features = model_options.get("cat_features")
        if cat_features is None:
            # Detecta automaticamente colunas que são do tipo 'category' ou 'object'
            cat_features = list(X_tr_res.select_dtypes(include=['category', 'object']).columns)
            logger.info(f"CatBoost: Features categóricas detectadas automaticamente: {cat_features}")
        
        # Converte colunas categóricas para o tipo 'category' se não forem
        for col in cat_features:
            X_tr_res[col] = X_tr_res[col].astype('category')
            X_val[col] = X_val[col].astype('category')

        model.fit(X_tr_res, y_tr_res, cat_features=cat_features)
    else:
        model.fit(X_tr_res, y_tr_res)

    # Log de métricas de validação
    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, zero_division=0)
    logger.info(f"Métricas de validação para {model_type} -> Acurácia: {acc:.3f} | F1-Score: {f1:.3f}")

    return model

def predict_espera(
    model: Any, X_to_predict: pd.DataFrame
) -> np.ndarray:
    """Gera predições de 'espera' para o dataset de submissão."""
    # Tratamento para CatBoost
    if hasattr(model, 'get_cat_feature_indices'):
        cat_features_indices = model.get_cat_feature_indices()
        cat_feature_names = [X_to_predict.columns[i] for i in cat_features_indices]
        for col in cat_feature_names:
            X_to_predict[col] = X_to_predict[col].astype('category')

    return model.predict(X_to_predict)

def build_submission(
    flightid: pd.Series, espera_pred: np.ndarray
) -> pd.DataFrame:
    """Cria o DataFrame de submissão final."""
    submission = pd.DataFrame({"flightid": flightid, "espera": espera_pred.astype(int)})
    return submission