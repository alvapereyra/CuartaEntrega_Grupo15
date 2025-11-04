import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np
import datetime

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis y Predicción de Posiciones NBA",
    page_icon="🏀",
    layout="wide"
)

# --- FUNCIONES DE TRANSFORMACIÓN DEL PIPELINE ---
# Estas funciones DEBEN estar definidas en el script para que
# joblib.load() pueda encontrar las referencias del pipeline.

def calculate_age(df):
    """
    Toma un array de NumPy con una columna 'birthdate', 
    la convierte a edad y la retorna como un array 2D.
    """
    # El pipeline pasa un array de NumPy
    birthdate = pd.to_datetime(df[:, 0], errors='coerce')
    
    # Fecha de la consigna
    today = datetime.datetime(2025, 10, 22)
    
    # El resultado (today - birthdate) es un TimedeltaIndex.
    # Se accede a los días con .days, no con .dt.days
    age = (today - birthdate).days / 365.25
    
    # Retorna como array 2D para que el pipeline lo entienda
    return age.values.reshape(-1, 1) 

def calculate_stats_36min(df):
    """
    Toma un array de NumPy con las 8 columnas de 'features_stats',
    calcula las métricas "por 36 minutos" (incluyendo Pts 2/3)
    y las retorna como un array 2D.
    """
    # El pipeline pasa un array de NumPy. Lo convertimos de nuevo a
    # DataFrame con los nombres de columna correctos para poder procesarlo.
    column_names = ['points', 'numMinutes', 'reboundsTotal', 'blocks', 'assists', 'steals', 'threePointersMade', 'freeThrowsMade']
    stats = pd.DataFrame(df, columns=column_names)

    # Evitar división por cero
    mask = stats["numMinutes"] > 0
    factor = np.where(mask, 36.0 / stats["numMinutes"], 0) 
    
    # --- Dividir Puntos --- 
    pts_from_3 = stats["threePointersMade"] * 3
    pts_from_2 = stats["points"] - pts_from_3 - stats["freeThrowsMade"]
    pts_from_2 = np.clip(pts_from_2, 0, None) 
    
    # --- Aplicar el factor de 36 minutos --- 
    out_df = pd.DataFrame(index=stats.index)
    out_df["reb36Min"] = stats["reboundsTotal"] * factor
    out_df["blk36Min"] = stats["blocks"] * factor
    out_df["ast36Min"] = stats["assists"] * factor
    out_df["stl36Min"] = stats["steals"] * factor
    out_df["pts_from_2_36Min"] = pts_from_2 * factor
    out_df["pts_from_3_36Min"] = pts_from_3 * factor
    
    return out_df.values

# -----------------------------------------------

# --- Carga de Activos (Modelo y Datos) ---

@st.cache_resource
def load_model():
    """Carga el modelo de predicción (pipeline) desde el archivo .pkl"""
    try:
        # Este archivo fue generado por la Entrega 3
        model = joblib.load("nba_position_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Error: Archivo 'nba_position_model.pkl' no encontrado.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_data
def load_data():
    """Carga el DataFrame limpio para las visualizaciones"""
    try:
        # Este archivo fue generado al final de la Entrega 3
        df = pd.read_csv("players_nba_clean_viz.csv")
        return df
    except FileNotFoundError:
        st.error("Error: Archivo 'players_nba_clean_viz.csv' no encontrado.")
        return pd.DataFrame() # Retorna DF vacío para evitar más errores
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

model = load_model()
df_viz = load_data()

# --- Definición de Columnas (para el formulario) ---
# Estas deben coincidir EXACTAMENTE con las 'ALL_FEATURES' del notebook de la Entrega 3
features_num_simple = ['height', 'bodyWeight']
features_cat = ['country']
features_age = ['birthdate']
features_stats = [
    'points', 'numMinutes', 'reboundsTotal', 'blocks', 
    'assists', 'steals', 'threePointersMade', 'freeThrowsMade'
]
ALL_FEATURES = features_num_simple + features_cat + features_age + features_stats

# Obtenemos una lista de países únicos del CSV de la Entrega 1
# Por ahora, usamos una lista hardcodeada con los más comunes.
COUNTRIES = [
    'USA', 'Canada', 'France', 'Spain', 'Serbia', 'Germany', 'Australia', 
    'Nigeria', 'Turkey', 'Brazil', 'Argentina', 'Lithuania', 'Otro'
]


# --- Sidebar (Interfaz de Predicción) ---
# Esta sección cumple con la consigna de "ofrecer una interfaz sencilla 
# para que un usuario final pueda ingresar datos nuevos y probar el modelo entrenado"

st.sidebar.image("https://cdn.freebiesupply.com/images/large/2x/nba-logo-transparent.png", width=150)
st.sidebar.title("Predecir Posición de Jugador")
st.sidebar.markdown("Ingresa los atributos (promedios por partido) de un jugador para predecir su posición (G, F o C).")

# Creamos un diccionario para guardar los inputs
user_input = {}

st.sidebar.header("Atributos Físicos")
user_input['birthdate'] = st.sidebar.date_input("Fecha de Nacimiento", 
                                                datetime.date(1998, 1, 1),
                                                min_value=datetime.date(1970, 1, 1),
                                                max_value=datetime.date(2008, 1, 1))
# Corregimos las etiquetas a pulgadas y libras
user_input['height'] = st.sidebar.slider("Altura (pulgadas)", 70, 90, 77)
user_input['bodyWeight'] = st.sidebar.slider("Peso (libras)", 160, 320, 210)
user_input['country'] = st.sidebar.selectbox("País", COUNTRIES, index=COUNTRIES.index('USA'))

st.sidebar.header("Estadísticas (Promedios por Partido)")
col1, col2 = st.sidebar.columns(2)
user_input['numMinutes'] = col1.slider("Minutos", 0.0, 40.0, 25.0, 0.1)
user_input['points'] = col2.slider("Puntos", 0.0, 40.0, 15.0, 0.1)
user_input['reboundsTotal'] = col1.slider("Rebotes", 0.0, 15.0, 5.0, 0.1)
user_input['assists'] = col2.slider("Asistencias", 0.0, 12.0, 3.0, 0.1)
user_input['blocks'] = col1.slider("Bloqueos", 0.0, 4.0, 0.5, 0.1)
user_input['steals'] = col2.slider("Robos", 0.0, 3.0, 0.8, 0.1)
user_input['threePointersMade'] = col1.slider("Triples Metidos", 0.0, 5.0, 1.5, 0.1)
user_input['freeThrowsMade'] = col2.slider("Libres Metidos", 0.0, 10.0, 3.0, 0.1)

# Botón de predicción
if st.sidebar.button("Predecir Posición", use_container_width=True, type="primary"):
    if model is None:
        st.error("El modelo no está cargado. No se puede predecir.")
    else:
        try:
            # 1. Convertir 'birthdate' (date) a Timestamp de pandas (que espera el pipeline)
            user_input['birthdate'] = pd.to_datetime(user_input['birthdate'])
            
            # 2. Crear DataFrame de 1 fila
            user_df = pd.DataFrame([user_input])
            
            # 3. Asegurar el orden de las columnas (¡CRUCIAL!)
            user_df = user_df[ALL_FEATURES]
            
            # 4. Hacer la predicción
            prediction = model.predict(user_df)
            probability = model.predict_proba(user_df)
            
            # 5. Mostrar resultados
            pos_predicha = prediction[0]
            prob_max = np.max(probability)
            
            st.sidebar.subheader(f"Predicción: ¡{pos_predicha}!")
            if pos_predicha == 'G':
                st.sidebar.info("🏀 **Guardia (G)**: Jugador enfocado en asistencias, robos y puntos de triple.")
            elif pos_predicha == 'F':
                st.sidebar.info("🔥 **Alero (F)**: Jugador versátil, balanceado entre anotación y rebotes.")
            elif pos_predicha == 'C':
                st.sidebar.info("🛡️ **Pivot (C)**: Jugador grande enfocado en rebotes, bloqueos y altura.")
            
            st.sidebar.write(f"**Confianza de la predicción:** `{prob_max:.1%}`")
            
            # Mostrar probabilidades detalladas
            prob_df = pd.DataFrame(probability, columns=model.classes_, index=["Probabilidad"])
            st.sidebar.dataframe(prob_df.style.format("{:.1%}"))

        except Exception as e:
            st.sidebar.error(f"Error al predecir: {e}")
            st.sidebar.write("Asegúrate de que 'nba_position_model.pkl' esté actualizado.")


# --- Cuerpo Principal de la App (Visualizaciones) ---
# Esta sección cumple con la consigna de "desarrollar una aplicación en Streamlit 
# que permita explorar los datos y resultados visualizados"

st.title("🏀 Análisis de Jugadores y Predicción de Posiciones NBA")
st.markdown(f"""
Esta aplicación presenta los hallazgos de las Entregas 3 y 4 (Grupo 15).
Utiliza un modelo `RandomForestClassifier` (F1-Score: **83.0%**) 
entrenado con datos de jugadores de la NBA para predecir su posición (`G`, `F`, `C`).
El set de datos para visualización contiene **{len(df_viz)}** jugadores.
""")

# --- Pestañas para las visualizaciones ---
tab1, tab2, tab3 = st.tabs([
    "📊 Gráfico 1: Perfil de Rol (Boxplots)", 
    "📈 Gráfico 2: Perfil Físico (Heatmap)", 
    "🏆 Gráfico 3: Importancia de Features"
])

if df_viz.empty:
    st.error("No se pudieron cargar los datos para las visualizaciones.")
else:
    with tab1:
        st.header("Gráfico 1: Perfil de Rol (Asistencias vs. Rebotes)")
        st.markdown("""
        Este gráfico justifica por qué el modelo puede separar las posiciones.
        * **Asistencias:** Las cajas no se solapan. Los **Guardias (G)** son una clase aparte en *playmaking*.
        * **Rebotes:** Vemos la relación inversa. Los **Pivots (C)** dominan la pintura, seguidos por los Aleros (F).
        """)
        
        # --- Chart 1: Boxplots (de la Entrega 4) ---
        # Gráfico 1a: Distribución de Asistencias
        chart_ast = alt.Chart(df_viz).mark_boxplot().encode(
            x=alt.X('dom_pos', title='Posición', sort=['G', 'F', 'C']),
            y=alt.Y('ast36Min', title='Asistencias por 36 min'),
            color=alt.Color('dom_pos', title='Posición', legend=None,
                            scale=alt.Scale(domain=['G', 'F', 'C'], range=['#1f77b4', '#ff7f0e', '#2ca02c'])),
            tooltip=['dom_pos', alt.Tooltip('ast36Min', title='Mediana Asistencias', format='.2f')]
        ).properties(
            title='Rol: Playmaking (Asistencias)'
        )
        # Gráfico 1b: Distribución de Rebotes
        chart_reb = alt.Chart(df_viz).mark_boxplot().encode(
            x=alt.X('dom_pos', title='Posición', sort=['G', 'F', 'C']),
            y=alt.Y('reb36Min', title='Rebotes por 36 min'),
            color=alt.Color('dom_pos', title='Posición', legend=None,
                            scale=alt.Scale(domain=['G', 'F', 'C'], range=['#1f77b4', '#ff7f0e', '#2ca02c'])),
            tooltip=['dom_pos', alt.Tooltip('reb36Min', title='Mediana Rebotes', format='.2f')]
        ).properties(
            title='Rol: Presencia en Pintura (Rebotes)'
        )
        final_chart1 = alt.hconcat(chart_ast, chart_reb)
        
        st.altair_chart(final_chart1, use_container_width=True)

    with tab2:
        st.header("Gráfico 2: Perfil Físico (Altura vs. Peso)")
        st.markdown("""
        Este heatmap facetado muestra las "huellas" físicas de cada posición.
        Vemos 3 *clusters* claros que el modelo usa para predecir:
        * **Guardias (G):** Cluster en la esquina inferior-izquierda (bajos y ligeros).
        * **Pivots (C):** Cluster en la esquina superior-derecha (altos y pesados).
        * **Aleros (F):** Cluster más disperso, ocupando el centro.
        """)
        
        # --- Chart 2: Heatmap Facetado (de la Entrega 4) ---
        chart2 = alt.Chart(df_viz).mark_rect().encode(
            x=alt.X('height', 
                  bin=alt.Bin(maxbins=20), 
                  title='Altura (pulgadas)'
                 ),
            y=alt.Y('bodyWeight', 
                  bin=alt.Bin(maxbins=20), 
                  title='Peso (libras)'
                 ),
            color=alt.Color('count()', title='Concentración', scale=alt.Scale(range='heatmap')),
            tooltip=['count()']
        ).properties(
            title='Hallazgo 2: Perfil Físico (Dónde se concentra cada posición)'
        ).facet(
            column=alt.Column('dom_pos', title='Posición Dominante', sort=['G', 'F', 'C'])
        ).interactive()
        
        st.altair_chart(chart2, use_container_width=True)
        
    with tab3:
        st.header("Gráfico 3: ¿Qué Features usa el Modelo para Predecir?")
        st.markdown("""
        Este gráfico confirma que las **`height`, `reb36Min`, `bodyWeight` y `ast36Min`** son las variables más decisivas 
        para el modelo de Random Forest.
        """)
        
        # --- Chart 3: Importancia de Features (de la Entrega 4) ---
        # Recreamos el DataFrame de importancia de features
        feature_data = {
            'Feature': [
                'height', 'reb36Min', 'bodyWeight', 'ast36Min', 'blk36Min',
                'age', 'pts_from_2_36Min', 'pts_from_3_36Min', 'stl36Min'
            ],
            'Importance': [
                0.241873, 0.133539, 0.125515, 0.110660, 0.085263,
                0.082465, 0.066028, 0.065390, 0.064831
            ]
        }
        df_imp = pd.DataFrame(feature_data)
        
        chart3 = alt.Chart(df_imp).mark_bar().encode(
            y=alt.Y('Feature', sort='-x'),
            x=alt.X('Importance', title='Importancia (Gini)'),
            tooltip=['Feature', 'Importance']
        ).properties(
            title='Importancia de Features del Modelo Final'
        )
        
        st.altair_chart(chart3, use_container_width=True)