import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def parse_wind_info(wind_info):
    direction_speed = wind_info.split(" at ")

    wind_direction = direction_speed[0]

    wind_speed = direction_speed[1]

    wind_speed = wind_speed.replace(" knots", "")

    return wind_direction, np.float64(wind_speed)

def get_metar_features(metar):
    features = {}
    type_of_clouds = ["few", "broken", "overcast", "scattered"]

    # Append time
    features['time'] = metar.time.ctime() or None

    # Append temperature
    features['temperature'] = metar.temp.string("C").split()[0] if metar.temp.string("C") else None

    # Append dew point
    features['dew_point'] = metar.dewpt.string("C").split()[0] if metar.dewpt.string("C") else None

    # Append wind direction and velocity
    if metar.wind():
        direction, velocity = parse_wind_info(metar.wind())
        features['wind_direction'] = direction
        features['wind_velocity'] = velocity
    else:
        features['wind_direction'] = None
        features['wind_velocity'] = None

    # Append peak wind
    features['peak_wind'] = metar.peak_wind() if metar.peak_wind() not in [None, "missing"] else None

    # Append wind shift
    features['wind_shift'] = metar.wind_shift() if metar.wind_shift() not in [None, "missing"] else None

    # Append visibility
    features['visibility'] = metar.visibility("M").split(' than ')[1].split()[0] if metar.visibility("M") else None

    # Append runway visual range
    features['runway_visual_range'] = metar.runway_visual_range("M").split()[0] if metar.runway_visual_range("M") else None

    # Append pressure
    features['pressure'] = metar.press.string("mb").split()[0] if metar.press.string("mb") else None

    # Append present weather
    features['present_weather'] = metar.present_weather() or None

    # Append sky conditions
    if metar.sky_conditions():
        conditions = metar.sky_conditions().split(';')
        for cloud in type_of_clouds:
            cloud_cond = next((cond for cond in conditions if cloud in cond), None)
            features[cloud + '_cloud_height'] = cloud_cond.split(" at ")[1].split()[0] if cloud_cond else None
    else:
        for cloud in type_of_clouds:
            features[cloud + '_cloud_height'] = None

    # Append temperature extremes and precipitation
    for attr in ['max_temp_6hr', 'min_temp_6hr', 'max_temp_24hr', 'min_temp_24hr',
                 'precip_1hr', 'precip_3hr', 'precip_6hr', 'precip_24hr',
                 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr']:
        features[attr] = getattr(metar, attr, None)

    return features


def get_voos_para_o_destino_no_mesmo_horario(destino, voos):
    data_hora = datetime.fromisoformat(destino['hora_ref'])
    data_hora_menos_uma_hora = data_hora - timedelta(hours=1)
    data_hora_mais_uma_hora = data_hora + timedelta(hours=1)


    data_hora_menos_uma_hora_str = data_hora_menos_uma_hora.isoformat()
    data_hora_mais_uma_hora_str = data_hora_mais_uma_hora.isoformat()

    #   Criar uma variavel com todos as linhas nos horarios acima
    voos_proximos = voos[(voos['hora_ref'] == data_hora_menos_uma_hora_str) | (voos['hora_ref'] == data_hora_mais_uma_hora_str) | (voos['hora_ref'] == destino['hora_ref'])]

    #   Retornar  a quantidade de voos em inteiro para a mesma origem do destino no voo proximo

    return voos_proximos[voos_proximos['origem'] == destino['origem']].shape[0]



