import pandas as pd
import numpy as np
import re
from metar import Metar
from datetime import datetime, timedelta
from geopy.distance import geodesic


def _parse_metar_string(metar_code: str) -> dict:
    """Parses a METAR string and extracts relevant features."""
    data = {}
    if pd.isna(metar_code) or not isinstance(metar_code, str):
        return {
            'station': None, 'report_time': None, 'temperature': None,
            'dew_point': None, 'wind_direction': None, 'wind_speed': None,
            'visibility': None, 'pressure': None, 'sky_conditions': None,
            'clouds_few': None, 'clouds_sct': None, 'clouds_bkn': None, 'clouds_ovc': None,
            'cloud_base_alt': None
        }
    try:
        obs = Metar.Metar(metar_code)
        data['station'] = obs.station_id
        data['report_time'] = obs.time.isoformat() if obs.time else None
        data['temperature'] = obs.temp.value() if obs.temp else None
        data['dew_point'] = obs.dewpt.value() if obs.dewpt else None

        if obs.wind_dir:
            data['wind_direction'] = obs.wind_dir.value()
        else:
            data['wind_direction'] = None
        data['wind_speed'] = obs.wind_speed.value() if obs.wind_speed else None

        data['visibility'] = obs.visibility.value() if obs.visibility else None
        data['pressure'] = obs.press.value() if obs.press else None
        data['sky_conditions'] = obs.sky_conditions()

        # Cloud coverage and altitude
        data['clouds_few'] = 0
        data['clouds_sct'] = 0
        data['clouds_bkn'] = 0
        data['clouds_ovc'] = 0
        data['cloud_base_alt'] = np.nan

        if obs.sky:
            for sky_cond in obs.sky:
                if sky_cond.cover == "FEW":
                    data['clouds_few'] = 1
                elif sky_cond.cover == "SCT":
                    data['clouds_sct'] = 1
                elif sky_cond.cover == "BKN":
                    data['clouds_bkn'] = 1
                elif sky_cond.cover == "OVC":
                    data['clouds_ovc'] = 1
                if sky_cond.base:
                    data['cloud_base_alt'] = sky_cond.base * 100 # in feet

    except Exception as e:
        # print(f"Error parsing METAR '{metar_code}': {e}")
        return {
            'station': None, 'report_time': None, 'temperature': None,
            'dew_point': None, 'wind_direction': None, 'wind_speed': None,
            'visibility': None, 'pressure': None, 'sky_conditions': None,
            'clouds_few': None, 'clouds_sct': None, 'clouds_bkn': None, 'clouds_ovc': None,
            'cloud_base_alt': None
        }
    return data

def _parse_taf_string(taf_code: str) -> dict:
    """Parses a simplified TAF string and extracts relevant features.
    This is a simplified parser as Metar library does not fully support TAF.
    """
    data = {}
    if pd.isna(taf_code) or not isinstance(taf_code, str):
        return {
            'station': None, 'forecast_time': None, 'wind_direction': None,
            'wind_speed': None, 'visibility': None, 'weather': None,
            'sky_conditions': None, 'temperature': None, 'dew_point': None
        }
    try:
        parts = taf_code.split()
        data['station'] = parts[1] if len(parts) > 1 else None
        data['forecast_time'] = parts[2] if len(parts) > 2 else None

        # Wind
        wind_match = re.search(r'(VRB|\d{3})(\d{2,3})KT', taf_code)
        if wind_match:
            data['wind_direction'] = int(wind_match.group(1)) if wind_match.group(1) != 'VRB' else np.nan
            data['wind_speed'] = int(wind_match.group(2))
        else:
            data['wind_direction'] = None
            data['wind_speed'] = None

        # Visibility
        vis_match = re.search(r' (\d{4}) ', taf_code)
        if vis_match:
            data['visibility'] = int(vis_match.group(1))
        elif 'CAVOK' in taf_code:
            data['visibility'] = 9999
        else:
            data['visibility'] = None

        # Weather (simplified)
        weather_patterns = ['TS', 'RA', 'SN', 'FG', 'BR', 'HZ'] # Add more as needed
        found_weather = [p for p in weather_patterns if p in taf_code]
        data['weather'] = ', '.join(found_weather) if found_weather else None

        # Sky conditions (simplified)
        sky_match = re.search(r'(FEW|SCT|BKN|OVC)(\d{3})', taf_code)
        data['sky_conditions'] = sky_match.group(0) if sky_match else None

        # Temperature and Dew Point (simplified, often not in TAFs or in different format)
        temp_dew_match = re.search(r'T(M?\d{2})/(M?\d{2})', taf_code)
        if temp_dew_match:
            data['temperature'] = -int(temp_dew_match.group(1).replace('M', '')) if 'M' in temp_dew_match.group(1) else int(temp_dew_match.group(1))
            data['dew_point'] = -int(temp_dew_match.group(2).replace('M', '')) if 'M' in temp_dew_match.group(2) else int(temp_dew_match.group(2))
        else:
            data['temperature'] = None
            data['dew_point'] = None

    except Exception as e:
        # print(f"Error parsing TAF '{taf_code}': {e}")
        return {
            'station': None, 'forecast_time': None, 'wind_direction': None,
            'wind_speed': None, 'visibility': None, 'weather': None,
            'sky_conditions': None, 'temperature': None, 'dew_point': None
        }
    return data

def extract_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai features das colunas METAR e METAF usando a biblioteca Metar e regex."""
    metar_features = df['metar'].apply(lambda x: pd.Series(_parse_metar_string(x)))
    df = pd.concat([df, metar_features.add_prefix('metar_')], axis=1)

    metaf_features = df['metaf'].apply(lambda x: pd.Series(_parse_taf_string(x)))
    df = pd.concat([df, metaf_features.add_prefix('metaf_')], axis=1)

    return df

aeroportos = {
    'SBBR': (-15.869167, -47.920833),  # Brasília
    'SBCF': (-19.624166, -43.971943),  # Confins
    'SBCT': (-25.531944, -49.173611),  # Curitiba
    'SBFL': (-27.670556, -48.547222),  # Florianópolis
    'SBGL': (-22.810556, -43.250556),  # Galeão (Rio de Janeiro)
    'SBGR': (-23.435556, -46.473056),  # Guarulhos (São Paulo)
    'SBKP': (-23.008889, -47.134444),  # Viracopos (Campinas)
    'SBPA': (-29.993889, -51.171389),  # Porto Alegre
    'SBRF': (-8.126389, -34.924167),   # Recife
    'SBRJ': (-22.910556, -43.163056),  # Santos Dumont (Rio de Janeiro)
    'SBSP': (-23.626111, -46.656389),  # Congonhas (São Paulo)
    'SBSV': (-12.908611, -38.331944)   # Salvador
}

def _calculate_distance(aeroporto1: str, aeroporto2: str) -> float:
    """Calculates the geodesic distance between two airports in kilometers."""
    from geopy.distance import geodesic
    coords1 = aeroportos[aeroporto1]
    coords2 = aeroportos[aeroporto2]
    return geodesic(coords1, coords2).kilometers

def _find_nearest_airport_data(df: pd.DataFrame, current_row: pd.Series, target_col: str, time_window_hours: int = 1, max_distance_km: int = 250) -> str or None:
    """
    Finds the nearest available METAR/METAF data for a given airport and time.
    Searches within a time window and then by geographical proximity.
    """
    current_airport = current_row['destino']
    current_time = current_row['hora_ref']

    # 1. Search in the same airport within the time window
    time_filtered_df = df[
        (df['destino'] == current_airport) &
        (df['hora_ref'] >= current_time - timedelta(hours=time_window_hours)) &
        (df['hora_ref'] <= current_time + timedelta(hours=time_window_hours)) &
        (df[target_col].notna())
    ]
    if not time_filtered_df.empty:
        # Get the closest in time
        closest_row = time_filtered_df.iloc[(time_filtered_df['hora_ref'] - current_time).abs().argsort()[:1]]
        return closest_row[target_col].iloc[0]

    # 2. Search in nearby airports within the time window
    nearby_airports = []
    for airport_code, coords in aeroportos.items():
        if airport_code != current_airport and current_airport in aeroportos: # Ensure current_airport is in dict
            distance = _calculate_distance(current_airport, airport_code)
            if distance <= max_distance_km:
                nearby_airports.append((airport_code, distance))
    
    nearby_airports.sort(key=lambda x: x[1]) # Sort by distance

    for airport_code, _ in nearby_airports:
        time_filtered_df = df[
            (df['destino'] == airport_code) &
            (df['hora_ref'] >= current_time - timedelta(hours=time_window_hours)) &
            (df['hora_ref'] <= current_time + timedelta(hours=time_window_hours)) &
            (df[target_col].notna())
        ]
        if not time_filtered_df.empty:
            closest_row = time_filtered_df.iloc[(time_filtered_df['hora_ref'] - current_time).abs().argsort()[:1]]
            return closest_row[target_col].iloc[0]
    
    return None

def _impute_missing_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing METAR and METAF data using time and geographical proximity."""
    df_copy = df.copy()
    
    # Ensure hora_ref is datetime for calculations
    df_copy['hora_ref'] = pd.to_datetime(df_copy['hora_ref'], errors='coerce')

    for index, row in df_copy.iterrows():
        if pd.isna(row['metar']):
            imputed_metar = _find_nearest_airport_data(df_copy, row, 'metar')
            if imputed_metar:
                df_copy.loc[index, 'metar'] = imputed_metar
        
        if pd.isna(row['metaf']):
            imputed_metaf = _find_nearest_airport_data(df_copy, row, 'metaf')
            if imputed_metaf:
                df_copy.loc[index, 'metaf'] = imputed_metaf
    
    return df_copy

def _generate_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates statistical features for meteorological data over time windows."""
    df_copy = df.copy()
    
    # Ensure hora_ref is datetime and set as index for resampling
    df_copy['hora_ref'] = pd.to_datetime(df_copy['hora_ref'], errors='coerce')
    df_copy = df_copy.set_index('hora_ref')

    numeric_cols = [col for col in df_copy.columns if df_copy[col].dtype in [np.number]]
    
    # Define meteorological features to aggregate
    met_features = [
        'metar_temperature', 'metar_dew_point', 'metar_wind_speed', 'metar_visibility', 'metar_pressure',
        'metaf_temperature', 'metaf_dew_point', 'metaf_wind_speed', 'metaf_visibility',
        # Add other relevant numeric features extracted from METAR/TAF
    ]
    
    # Filter to only include existing numeric meteorological features
    met_features = [f for f in met_features if f in numeric_cols]

    time_windows = ['1H', '3H', '6H', '12H'] # 1 hour, 3 hours, 6 hours, 12 hours

    for airport in df_copy['destino'].unique():
        airport_df = df_copy[df_copy['destino'] == airport]
        
        for window in time_windows:
            # Resample and aggregate
            for feature in met_features:
                if feature in airport_df.columns:
                    # Mean
                    df_copy.loc[airport_df.index, f'mean_destino_over_{feature}_past_{window}'] = \
                        airport_df.groupby('destino')[feature].rolling(window=window, closed='left').mean().reset_index(level=0, drop=True)
                    # Min
                    df_copy.loc[airport_df.index, f'min_destino_over_{feature}_past_{window}'] = \
                        airport_df.groupby('destino')[feature].rolling(window=window, closed='left').min().reset_index(level=0, drop=True)
                    # Max
                    df_copy.loc[airport_df.index, f'max_destino_over_{feature}_past_{window}'] = \
                        airport_df.groupby('destino')[feature].rolling(window=window, closed='left').max().reset_index(level=0, drop=True)
                    # Count (non-NA)
                    df_copy.loc[airport_df.index, f'count_destino_over_{feature}_past_{window}'] = \
                        airport_df.groupby('destino')[feature].rolling(window=window, closed='left').count().reset_index(level=0, drop=True)
    
    df_copy = df_copy.reset_index() # Reset index to make hora_ref a column again
    return df_copy

def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara a matriz de features para o modelo."""
    if "hora_ref" in df.columns:
        df["hora_ref"] = pd.to_datetime(df["hora_ref"], errors="coerce")
        df["hora"] = df["hora_ref"].dt.hour
        df["dia"] = df["hora_ref"].dt.dayofyear

    df["precisa_troca"] = (
        df["prev_troca_cabeceira"].fillna(0).astype(int)
        | df["troca_cabeceira_hora_anterior"].fillna(0).astype(int)
    )

    df = pd.get_dummies(df, columns=["origem", "destino"], dtype=np.uint8)

    # Imputa valores nulos com a mediana
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

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

def preprocess_public_dataframe(public: pd.DataFrame, features_pca: pd.DataFrame):
    """Replica a limpeza e engenharia de features do notebook."""
    df = public.copy()
    df = extract_weather_features(df)
    df = df.drop_duplicates().reset_index(drop=True)

    df = pd.merge(df, features_pca, on="flightid", how="left")

    pca_cols = [col for col in df.columns if col.startswith('pca_')]
    for col in pca_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    df_train = df[df["espera"].notna()].copy()
    df_pred = df[df["espera"].isna()].copy()

    X_train = _prepare_feature_matrix(df_train)
    y_train = df_train["espera"].astype(int)
    X_to_predict = _prepare_feature_matrix(df_pred)

    X_train, X_to_predict = X_train.align(X_to_predict, join='left', axis=1, fill_value=0)

    flightid_pred = pd.Series(df_pred["flightid"].reset_index(drop=True))

    print("Pre-processamento concluido:")
    print("   – X_train shape:", X_train.shape)
    print("   – X_to_predict shape:", X_to_predict.shape)

    return X_train, y_train, X_to_predict, flightid_pred

def preprocess_public_dataframe_sem_pca(public: pd.DataFrame):
    """Replica a limpeza e engenharia de features do notebook sem PCA."""
    df = public.copy()
    df = extract_weather_features(df)
    df = _impute_missing_weather_data(df)
    df = _generate_statistical_features(df)
    df = df.drop_duplicates().reset_index(drop=True)

    df_train = df[df["espera"].notna()].copy()
    df_pred = df[df["espera"].isna()].copy()

    X_train = _prepare_feature_matrix(df_train)
    y_train = df_train["espera"].astype(int)
    X_to_predict = _prepare_feature_matrix(df_pred)

    X_train, X_to_predict = X_train.align(X_to_predict, join='left', axis=1, fill_value=0)

    flightid_pred = pd.Series(df_pred["flightid"].reset_index(drop=True))

    print("Pre-processamento concluido:")
    print("   – X_train shape:", X_train.shape)
    print("   – X_to_predict shape:", X_to_predict.shape)

    return X_train, y_train, X_to_predict, flightid_pred