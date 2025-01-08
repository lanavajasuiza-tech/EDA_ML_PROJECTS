import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import requests
import plotly.express as px
import time
from dotenv import load_dotenv
import os




# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("CMC_API_KEY")

# Configurar Streamlit
st.set_page_config(page_title="Bitcoin y Ethereum (Web3)", page_icon="📈", initial_sidebar_state="collapsed", layout="wide")

# API details for CoinMarketCap
base_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
headers = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY
}
params = {
    "id": "1,1027",  # IDs de Bitcoin y Ethereum
    "convert": "USD"
}

# Configuración de la base de datos
conn = sqlite3.connect("crypto_data.db")
cursor = conn.cursor()

# Verificar y actualizar la tabla si faltan columnas
cursor.execute("PRAGMA table_info(prices)")
existing_columns = [info[1] for info in cursor.fetchall()]
required_columns = [
    "time", "btc_price", "eth_price", "btc_market_cap", "eth_market_cap",
    "btc_volume_24h", "eth_volume_24h", "btc_percent_change_24h", "eth_percent_change_24h",
    "btc_percent_change_1h", "eth_percent_change_1h", "btc_percent_change_7d", "eth_percent_change_7d"
]

for column in required_columns:
    if column not in existing_columns:
        cursor.execute(f"ALTER TABLE prices ADD COLUMN {column} REAL")
conn.commit()


# Función para obtener datos de CoinMarketCap API
@st.cache_data(ttl=300)
def get_crypto_data():
    try:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()["data"]
            btc_data = data.get("1", {}).get("quote", {}).get("USD", {})
            eth_data = data.get("1027", {}).get("quote", {}).get("USD", {})

            btc_price = btc_data.get("price", 0.0)
            eth_price = eth_data.get("price", 0.0)
            btc_market_cap = btc_data.get("market_cap", 0.0)
            eth_market_cap = eth_data.get("market_cap", 0.0)
            btc_volume_24h = btc_data.get("volume_24h", 0.0)
            eth_volume_24h = eth_data.get("volume_24h", 0.0)
            btc_percent_change_24h = btc_data.get("percent_change_24h", 0.0)
            eth_percent_change_24h = eth_data.get("percent_change_24h", 0.0)
            btc_percent_change_1h = btc_data.get("percent_change_1h", 0.0)
            eth_percent_change_1h = eth_data.get("percent_change_1h", 0.0)
            btc_percent_change_7d = btc_data.get("percent_change_7d", 0.0)
            eth_percent_change_7d = eth_data.get("percent_change_7d", 0.0)

            return btc_price, eth_price, btc_market_cap, eth_market_cap, btc_volume_24h, eth_volume_24h, btc_percent_change_24h, eth_percent_change_24h, btc_percent_change_1h, eth_percent_change_1h, btc_percent_change_7d, eth_percent_change_7d
        else:
            print("Error al obtener datos de la API de CoinMarketCap")
            return [None] * 12
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        return [None] * 12
    


# Función para guardar datos si hay cambios
def save_data_if_changed(btc_price, eth_price, btc_market_cap, eth_market_cap, btc_volume_24h, eth_volume_24h, btc_percent_change_24h, eth_percent_change_24h, btc_percent_change_1h, eth_percent_change_1h, btc_percent_change_7d, eth_percent_change_7d):
    last_row = pd.read_sql_query("SELECT * FROM prices ORDER BY time DESC LIMIT 1", conn)

    if not last_row.empty:
        last_btc_price = last_row.iloc[0]['btc_price']
        last_eth_price = last_row.iloc[0]['eth_price']
        last_btc_market_cap = last_row.iloc[0]['btc_market_cap']
        last_eth_market_cap = last_row.iloc[0]['eth_market_cap']

        if (btc_price == last_btc_price and eth_price == last_eth_price and
                btc_market_cap == last_btc_market_cap and eth_market_cap == last_eth_market_cap):
            return False  # No hay cambios

    current_time = datetime.now()
    cursor.execute("INSERT INTO prices (time, btc_price, eth_price, btc_market_cap, eth_market_cap, btc_volume_24h, eth_volume_24h, btc_percent_change_24h, eth_percent_change_24h, btc_percent_change_1h, eth_percent_change_1h, btc_percent_change_7d, eth_percent_change_7d) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (current_time, btc_price, eth_price, btc_market_cap, eth_market_cap, btc_volume_24h, eth_volume_24h, btc_percent_change_24h, eth_percent_change_24h, btc_percent_change_1h, eth_percent_change_1h, btc_percent_change_7d, eth_percent_change_7d))
    conn.commit()
    return True  # Datos guardados

# Obtener datos y guardar en la base de datos
crypto_data = get_crypto_data()
if all(value is not None for value in crypto_data):
    save_data_if_changed(*crypto_data)

# Interfaz de Streamlit
st.title("Bitcoin vs Ethereum")

# Botón de actualización manual
if st.markdown("<style>.stButton>button{background-color:#007BFF;color:white;border-radius:5px;padding:10px;font-weight:bold;}</style>", unsafe_allow_html=True):
    st.markdown("<style>.stButton>button{background-color:#1abc9c;color:black;border-radius:5px;padding:10px;font-weight:bold;margin-right:10px;}</style>", unsafe_allow_html=True)
st.button("Actualizar Datos")
crypto_data = get_crypto_data()
if all(value is not None for value in crypto_data):
    save_data_if_changed(*crypto_data)
    st.query_params.from_dict({"refresh": "true"})

# Filtro de fechas
st.sidebar.subheader("Filtrar por fecha")
start_date = st.sidebar.date_input("Fecha de inicio", value=datetime.now().date())
end_date = st.sidebar.date_input("Fecha de fin", value=datetime.now().date())

# Mostrar datos y gráficos en columnas
col1, col2 = st.columns(2)

# Datos y gráfico de Bitcoin
with col1:
    st.subheader("Bitcoin (BTC)")
    df_btc = pd.read_sql_query("SELECT time, btc_price, btc_market_cap, btc_volume_24h, btc_percent_change_24h, btc_percent_change_1h, btc_percent_change_7d FROM prices", conn)
    if not df_btc.empty:
        df_btc['time'] = pd.to_datetime(df_btc['time'], errors='coerce')
        df_btc.dropna(subset=['time'], inplace=True)
        df_btc = df_btc[(df_btc['time'].dt.date >= start_date) & (df_btc['time'].dt.date <= end_date)]

        st.write("Precio actual: ${:.2f}".format(df_btc['btc_price'].iloc[-1]))
        st.write("Capitalización de mercado: ${:.2f} B USD".format(df_btc['btc_market_cap'].iloc[-1] / 1e9))
        st.write("Volumen 24h: ${:.2f} B USD".format(df_btc['btc_volume_24h'].iloc[-1] / 1e9))
        st.write(f"{'🟢' if df_btc['btc_percent_change_1h'].iloc[-1] >= 0 else '🔴'} Cambio 1h: {df_btc['btc_percent_change_1h'].iloc[-1]:.2f}%")
        st.write(f"{'🟢' if df_btc['btc_percent_change_24h'].iloc[-1] >= 0 else '🔴'} Cambio 24h: {df_btc['btc_percent_change_24h'].iloc[-1]:.2f}%")
        st.write(f"{'🟢' if df_btc['btc_percent_change_7d'].iloc[-1] >= 0 else '🔴'} Cambio 7d: {df_btc['btc_percent_change_7d'].iloc[-1]:.2f}%")

        fig_btc = px.line(df_btc, x='time', y='btc_price', title="Evolución del precio de Bitcoin",
                          labels={"time": "Hora", "btc_price": "Precio (USD)"},
                          color_discrete_sequence=["orange"])
        st.plotly_chart(fig_btc, use_container_width=True, key="btc_chart")

# Datos y gráfico de Ethereum
with col2:
    st.subheader("Ethereum (ETH)")
    df_eth = pd.read_sql_query("SELECT time, eth_price, eth_market_cap, eth_volume_24h, eth_percent_change_24h, eth_percent_change_1h, eth_percent_change_7d FROM prices", conn)
    if not df_eth.empty:
        df_eth['time'] = pd.to_datetime(df_eth['time'], errors='coerce')
        df_eth.dropna(subset=['time'], inplace=True)
        df_eth = df_eth[(df_eth['time'].dt.date >= start_date) & (df_eth['time'].dt.date <= end_date)]

        st.write("Precio actual: ${:.2f}".format(df_eth['eth_price'].iloc[-1]))
        st.write("Capitalización de mercado: ${:.2f} B USD".format(df_eth['eth_market_cap'].iloc[-1] / 1e9))
        st.write("Volumen 24h: ${:.2f} B USD".format(df_eth['eth_volume_24h'].iloc[-1] / 1e9))
        st.write(f"{'🟢' if df_eth['eth_percent_change_1h'].iloc[-1] >= 0 else '🔴'} Cambio 1h: {df_eth['eth_percent_change_1h'].iloc[-1]:.2f}%")
        st.write(f"{'🟢' if df_eth['eth_percent_change_24h'].iloc[-1] >= 0 else '🔴'} Cambio 24h: {df_eth['eth_percent_change_24h'].iloc[-1]:.2f}%")
        st.write(f"{'🟢' if df_eth['eth_percent_change_7d'].iloc[-1] >= 0 else '🔴'} Cambio 7d: {df_eth['eth_percent_change_7d'].iloc[-1]:.2f}%")

        fig_eth = px.line(df_eth, x='time', y='eth_price', title="Evolución del precio de Ethereum",
                          labels={"time": "Hora", "eth_price": "Precio (USD)"},
                          color_discrete_sequence=["lightgray"])
        st.plotly_chart(fig_eth, use_container_width=True, key="eth_chart")

# Exportar datos a CSV
st.sidebar.subheader("Exportar datos")
if st.sidebar.button("Exportar a CSV"):
    df_export = pd.read_sql_query("SELECT * FROM prices", conn)
    csv = df_export.to_csv(index=False)
    st.sidebar.download_button(label="Descargar CSV", data=csv, file_name="crypto_data.csv", mime="text/csv")

# Refrescar automáticamente cada 5 minutos
st.query_params.from_dict({"refresh": str(datetime.now())})
time.sleep(300)
st.experimental_rerun()


# Cerrar la conexión a la base de datos
conn.close()
