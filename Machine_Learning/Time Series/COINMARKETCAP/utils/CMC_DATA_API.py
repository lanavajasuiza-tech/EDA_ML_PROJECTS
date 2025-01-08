#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre del Script: CMC_DATA_API.py
Descripción: 
    Este script utiliza la API de CoinMarketCap, extraer información sobre criptomonedas y generar 
    archivos CSV y Excel con los datos completos y filtrados en la carpeta datos_origen

Autor: Ana Ndongo
Fecha: 2024-06-17
Versión: 1.0
Dependencias:
    - Requierements_market.txt
"""


import requests
import pandas as pd
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Clave de API desde el archivo .env
API_KEY = os.getenv("API_KEY")

# Tu clave de API
#API_KEY = "54b9c36a-9f32-44aa-9388-4215331c04bf"

# Endpoint base de la API
BASE_URL = "https://pro-api.coinmarketcap.com/v1/"

# Headers requeridos por la API
HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY
}

def get_cryptocurrency_data():
    """
    Consulta las criptomonedas más populares de CoinMarketCap.
    """
    url = BASE_URL + "cryptocurrency/listings/latest"
    params = {
        "start": 1,  # Desde la primera moneda
        "limit": 5000,  # Número de monedas a consultar
        "convert": "USD"  # Moneda para la conversión
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error en la consulta: {response.status_code} - {response.json().get('status', {}).get('error_message', 'Error desconocido')}")
            return None
    except Exception as e:
        print(f"Error en la conexión: {e}")
        return None

def process_full_crypto_data(data):
    """
    Procesa el JSON completo de la API y extrae toda la información relevante en un DataFrame.
    """
    crypto_list = []
    
    for crypto in data.get("data", []):  # Accede a la lista de criptomonedas
        # Información general
        crypto_data = {
            "id": crypto.get("id"),
            "name": crypto.get("name"),
            "symbol": crypto.get("symbol"),
            "num_market_pairs": crypto.get("num_market_pairs"),
            "date_added": crypto.get("date_added"),
            "max_supply": crypto.get("max_supply"),
            "circulating_supply": crypto.get("circulating_supply"),
            "total_supply": crypto.get("total_supply"),
            "infinite_supply": crypto.get("infinite_supply"),
            "cmc_rank": crypto.get("cmc_rank"),
            "slug": crypto.get("slug"),
            "last_updated": crypto.get("last_updated"),
            "tags": ", ".join(crypto.get("tags", []))  # Combina las etiquetas en un solo string
        }


        # Información de mercado (desde "quote")
        market_data = crypto.get("quote", {}).get("USD", {})
        crypto_data.update({
            "price": market_data.get("price"),
            "volume_24h": market_data.get("volume_24h"),
            "volume_change_24h": market_data.get("volume_change_24h"),
            "percent_change_1h": market_data.get("percent_change_1h"),
            "percent_change_24h": market_data.get("percent_change_24h"),
            "percent_change_7d": market_data.get("percent_change_7d"),
            "percent_change_30d": market_data.get("percent_change_30d"),
            "percent_change_60d": market_data.get("percent_change_60d"),
            "percent_change_90d": market_data.get("percent_change_90d"),
            "market_cap": market_data.get("market_cap"),
            "market_cap_dominance": market_data.get("market_cap_dominance"),
            "fully_diluted_market_cap": market_data.get("fully_diluted_market_cap"),
        })


        # Información de la plataforma (si existe)
        platform = crypto.get("platform", {})
        if platform:
            crypto_data.update({
                "platform_id": platform.get("id"),
                "platform_name": platform.get("name"),
                "platform_symbol": platform.get("symbol"),
                "platform_slug": platform.get("slug"),
                "platform_token_address": platform.get("token_address"),
            })
        else:
            crypto_data.update({
                "platform_id": None,
                "platform_name": None,
                "platform_symbol": None,
                "platform_slug": None,
                "platform_token_address": None,
            })

        # Añadimos la criptomoneda al listado
        crypto_list.append(crypto_data)

    # Convertimos la lista de diccionarios a un DataFrame
    crypto_df = pd.DataFrame(crypto_list)

   # Procesamos las etiquetas (tags) si están presentes, las tratamos en columnas con valores dummies para análsis predectivos
    if "tags" in crypto_df.columns:
        # Convertimos las etiquetas en listas y generamos las columnas binarias
        crypto_df["tags"] = crypto_df["tags"].str.split(", ")  # Convertimos a listas
        tags_dummies = crypto_df["tags"].explode().str.get_dummies().groupby(level=0).max()

        # Agregamos el prefijo 'tag_' a las columnas binarias
        tags_dummies = tags_dummies.add_prefix("tag_")

        # Combinamos las columnas binarias con el DataFrame original
        crypto_df = pd.concat([crypto_df, tags_dummies], axis=1)

        # Eliminamos la columna original "tags" para evitar redundancias
        crypto_df = crypto_df.drop(columns=["tags"])

    return crypto_df

def format_numbers_with_thousands_separator(df, columns):
    """
    Formatea las columnas numéricas para incluir separadores de miles.
    """
    for col in columns:
        if col in df.columns:
            # Aplicamos el formato para separadores de miles con `.`
            df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    return df

# Ejecución del script
if __name__ == "__main__":
    # Llamamos a la API para obtener datos
    crypto_data = get_cryptocurrency_data()
    if crypto_data:
        # Procesamos los datos en un DataFrame
        crypto_df = process_full_crypto_data(crypto_data)

        # Verificamos si hay datos en el DataFrame
        if not crypto_df.empty:
            # Guardamos los datos procesados en un archivo CSV
            crypto_df.to_csv(r"/home/ana/Documentos/SCRIPTS_PYTHON/COINMARKETCAP/datos_origen/crypto_full_data.csv", index=False)

            # Guardamos los datos procesados en un archivo Excel
            #crypto_df.to_excel(r"/home/ana/Documentos/SCRIPTS_PYTHON/COINMARKETCAP/datos_origen/crypto_full_data.xlsx", index=False, engine="openpyxl")

            print("Datos guardados en 'crypto_full_data.csv' y 'crypto_full_data.xlsx'")
        else:
            print("No se encontraron datos en la respuesta JSON.")
        # Procesamos los datos en un DataFrame
        crypto_df = process_full_crypto_data(crypto_data)

        # Aplicamos el formato con separadores de miles (opcional)
        numeric_columns = [
            "price", "market_cap", "volume_24h", "fully_diluted_market_cap",
            "max_supply", "circulating_supply", "total_supply"
        ]
        crypto_df = format_numbers_with_thousands_separator(crypto_df, numeric_columns)

        # Filtrar por las monedas de interés
        selected_coins = [
            "Bitcoin", "Ethereum", "Aave", "POL (ex-MATIC)", "TRON", 
            "IOTA", "Litecoin", "Basic Attention Token", "Shimmer", "BitTorrent [New]"
        ]
        filtered_crypto = crypto_df[crypto_df["name"].isin(selected_coins)]

        # Seleccionar columnas clave
        fields_of_interest = [
            "name", "symbol", "price", "market_cap", 
            "percent_change_24h", "percent_change_7d", "volume_24h"
        ]
        filtered_data = filtered_crypto[fields_of_interest]

        # Mostrar los datos filtrados
        print(filtered_data)

        # Guardar los datos filtrados en CSV y Excel
        filtered_data.to_csv(r"/home/ana/Documentos/SCRIPTS_PYTHON/COINMARKETCAP/datos_origen/filtered_crypto.csv", index=False)
        #filtered_data.to_excel(r"/home/ana/Documentos/SCRIPTS_PYTHON/COINMARKETCAP/datos_origen/filtered_crypto.xlsx", index=False, engine="openpyxl")

        print("Datos filtrados guardados en 'filtered_crypto.csv' y 'filtered_crypto.xlsx'.")
    else:
        print("No se encontraron datos en la respuesta JSON.")