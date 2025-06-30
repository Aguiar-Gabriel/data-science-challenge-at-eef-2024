import pandas as pd
import numpy as np

def load_public_dataset(public: pd.DataFrame) -> pd.DataFrame:
    print("✅ Dataset carregado. Shape:", public.shape)
    print("📌 Colunas:", public.columns.tolist())
    return public

def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the feature matrix by dropping non-numeric columns used only for description
    and converting datetime features.

    The logic follows exactly the transformations applied in the *project002.ipynb* notebook
    that serves as the source-of-truth for this project:

    1. Convert ``hora_ref`` to *datetime* (if still present).
    2. Extract ``hora`` (hour of day) and ``dia`` (day of year).
    3. Create boolean column ``precisa_troca`` as the logical *OR* between
       ``prev_troca_cabeceira`` and ``troca_cabeceira_hora_anterior``.
    4. One-hot encode the categorical airports columns ``origem`` and ``destino``.
    5. Drop columns that are not used as model inputs: *espera*, *flightid*,
       *url_img_satelite*, *metar*, *metaf*, *hora_ref* (after extraction).
    """

    # 1️⃣ Ensure datetime
    if "hora_ref" in df.columns:
        df["hora_ref"] = pd.to_datetime(df["hora_ref"], errors="coerce")

    # 2️⃣ Derive time based features
    if "hora_ref" in df.columns:
        df["hora"] = df["hora_ref"].dt.hour
        df["dia"] = df["hora_ref"].dt.dayofyear

    # 3️⃣ Domain engineered flag
    df["precisa_troca"] = (
        df["prev_troca_cabeceira"].fillna(0).astype(int)
        | df["troca_cabeceira_hora_anterior"].fillna(0).astype(int)
    )

    # 4️⃣ Categorical dummies
    df = pd.get_dummies(df, columns=["origem", "destino"], dtype=np.uint8)

    # 5️⃣ Drop non-feature columns
    to_drop = [
        "espera",
        "flightid",
        "url_img_satelite",
        "metar",
        "metaf",
        "hora_ref",
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    return df

def preprocess_public_dataframe(public: pd.DataFrame):  # noqa: D401 – simple name is fine
    """Replicate the cleaning and feature engineering performed in the notebook.

    Returns
    -------
    X_train : pd.DataFrame
        Feature matrix for rows where the target *espera* is **known**.
    y_train : pd.Series
        Binary target vector.
    X_to_predict : pd.DataFrame
        Feature matrix for rows where *espera* is **missing** and needs prediction.
    flightid_pred : pd.Series
        Corresponding *flightid* values for *X_to_predict* so they can be merged
        later into a submission file.
    """

    df = public.copy()

    # Basic missing-value handling exactly as notebook
    df["metaf"].fillna("DESCONHECIDO", inplace=True)

    # Remove duplicates just in case
    df = df.drop_duplicates().reset_index(drop=True)

    # Split into training and prediction partitions
    df_train = df[df["espera"].notna()].copy()
    df_pred = df[df["espera"].isna()].copy()

    # ------------------------------------------------------------------
    # Build feature matrices
    # ------------------------------------------------------------------
    X_train = _prepare_feature_matrix(df_train)  # type: ignore[arg-type]
    y_train = df_train["espera"].astype(int)

    X_to_predict = _prepare_feature_matrix(df_pred)  # type: ignore[arg-type]

    # Align columns (the notebook used pandas align with left join)
    X_train, X_to_predict = X_train.align(X_to_predict, join="left", axis=1, fill_value=0)

    flightid_pred = pd.Series(df_pred["flightid"].reset_index(drop=True))  # type: ignore[attr-defined]

    print("✅ Pre-processamento concluído:")
    print("   – X_train shape:", X_train.shape)
    print("   – X_to_predict shape:", X_to_predict.shape)

    return X_train, y_train, X_to_predict, flightid_pred
