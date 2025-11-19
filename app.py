import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np
import datetime

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="NBA Position Predictor - Grupo 15",
    page_icon="🏀",
    layout="wide"
)

# --- FUNCIONES HELPER DEL MODELO (NECESARIAS PARA JOBLIB) ---
# Estas funciones deben ser idénticas a las definidas en el notebook de entrenamiento V4

def calculate_age(df):
    # ... (esta función calculate_age déjala como estaba) ...
    try:
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
             birthdate = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        else:
             birthdate = pd.to_datetime(df[:, 0], errors='coerce')
             
        today = datetime.datetime(2025, 10, 22)
        age = (today - birthdate).dt.days / 365.25 if hasattr((today - birthdate), 'dt') else (today - birthdate).days / 365.25
        return age.values.reshape(-1, 1)
    except:
        return np.zeros((len(df), 1))

# --- CORRECCIÓN: CAMBIAR EL NOMBRE DE LA FUNCIÓN ---
# Antes se llamaba calculate_stats_36min_from_acum
# Ahora le ponemos el nombre que pide el error: calculate_stats_36min

def calculate_stats_36min_from_acum(df_acum):
    # Reconstruimos el DF temporalmente con los nombres esperados por la logica
    column_names = ['points_acum', 'numMinutes_acum', 'reboundsTotal_acum', 'blocks_acum', 
                    'assists_acum', 'steals_acum', 'threePointersMade_acum', 'fieldGoalsMade_acum']
    
    # Manejo de entrada (puede ser array numpy desde el pipeline)
    if isinstance(df_acum, np.ndarray):
        stats = pd.DataFrame(df_acum, columns=column_names)
    else:
        stats = df_acum.copy()
        stats.columns = column_names

    total_minutes = stats['numMinutes_acum']
    # Evitar división por 0
    total_minutes = total_minutes.replace(0, 1) 

    out_df = pd.DataFrame(index=stats.index)
    out_df["reb36Min"] = (stats["reboundsTotal_acum"] / total_minutes) * 36
    out_df["blk36Min"] = (stats["blocks_acum"] / total_minutes) * 36
    out_df["ast36Min"] = (stats["assists_acum"] / total_minutes) * 36
    out_df["stl36Min"] = (stats["steals_acum"] / total_minutes) * 36
    out_df["pts_from_3_36Min"] = (stats["threePointersMade_acum"] * 3 / total_minutes) * 36
    
    tiros_de_2_metidos = stats["fieldGoalsMade_acum"] - stats["threePointersMade_acum"]
    out_df["pts_from_2_36Min"] = (tiros_de_2_metidos * 2 / total_minutes) * 36
    
    out_df = out_df.clip(lower=0)
    
    return out_df.values

# --- CARGA DE DATOS Y MODELO ---

@st.cache_resource
def load_model():
    # Asegúrate de que 'nba_position_model.pkl' esté en la misma carpeta
    return joblib.load("nba_position_model.pkl")

@st.cache_data
def load_data():
    # Asegúrate de que 'players_nba_clean_viz.csv' esté en la misma carpeta
    return pd.read_csv("players_nba_clean_viz.csv")

try:
    model = load_model()
    df_viz = load_data()
except Exception as e:
    st.error(f"Error crítico cargando archivos: {e}")
    st.stop()

# --- PESTAÑAS DE LA APLICACIÓN ---
tab_info, tab_viz, tab_pred = st.tabs(["📘 Explicación del Modelo", "📊 Exploración de Datos", "🔮 Predicción Interactiva"])

# ==============================================================================
# PESTAÑA 1: EXPLICACIÓN DEL MODELO
# ==============================================================================
with tab_info:
    st.title("🏀 Clasificador de Posiciones NBA")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### El Problema
        En la NBA moderna, las posiciones tradicionales (Base, Alero, Pivot) son cada vez más difusas. 
        Nuestro objetivo fue entrenar un modelo de IA capaz de clasificar a un jugador basándose puramente 
        en sus **estadísticas de juego** y **atributos físicos**, sin ver su etiqueta oficial.

        ### El Modelo
        Utilizamos un **Random Forest Classifier** optimizado.
        * **Accuracy:** ~83%
        * **F1-Score:** ~81%
        
        El modelo aprende patrones complejos: entiende que un Pivot no es solo "alguien alto", 
        sino alguien que rebotea eficientemente y anota cerca del aro.
        """)
    
    with col2:
        st.info("💡 **Dato Curioso:** El error más común del modelo es confundir **Aleros (F)** con **Pivots (C)**, lo cual refleja la evolución de los 'Small-Ball Centers' en la liga.")

    st.divider()
    st.subheader("¿Qué mira el modelo? (Importancia de Features)")
    
    # Datos hardcodeados de la Entrega 3 para visualización rápida
    imp_data = pd.DataFrame({
        'Feature': ['Altura', 'Rebotes/36', 'Peso', 'Asistencias/36', 'Bloqueos/36', 'Edad', 'Pts Dobles', 'Pts Triples', 'Robos'],
        'Importancia': [0.24, 0.13, 0.12, 0.11, 0.085, 0.082, 0.066, 0.065, 0.064]
    })
    
    c_imp = alt.Chart(imp_data).mark_bar(color='#FF4B4B').encode(
        x=alt.X('Importancia', title='Peso en la decisión'),
        y=alt.Y('Feature', sort='-x', title='Atributo'),
        tooltip=['Feature', 'Importancia']
    ).properties(height=300)
    st.altair_chart(c_imp, use_container_width=True)


# ==============================================================================
# PESTAÑA 2: EXPLORACIÓN (GRAFICOS ENTREGA 4)
# ==============================================================================
with tab_viz:
    st.header("Explorando los Patrones")
    
    st.markdown("### 1. Roles de Juego: Asistencias vs. Rebotes")
    st.write("Este gráfico separa claramente a los creadores de juego (Guardias) de los definidores interiores (Pivots).")
    
    # Gráfico Boxplot Horizontal
    c_base = alt.Chart(df_viz).mark_boxplot(extent='min-max', size=30).encode(
        y=alt.Y('dom_pos', title=None),
        color=alt.Color('dom_pos', legend=None)
    ).properties(height=200)
    
    c1 = c_base.encode(x=alt.X('ast36Min', title='Asistencias p/36m')).properties(title='Distribución de Asistencias')
    c2 = c_base.encode(x=alt.X('reb36Min', title='Rebotes p/36m')).properties(title='Distribución de Rebotes')
    
    st.altair_chart(c1 | c2, use_container_width=True)
    
    st.divider()
    
    st.markdown("### 2. Perfil Físico y Técnico de cada posición")
    st.write("Usamos Coordenadas Paralelas para ver el perfil promedio de cada posición.")
    
    # Coordenadas Paralelas (Promedios)
    feats = ['height', 'bodyWeight', 'reb36Min', 'ast36Min', 'pts_from_3_36Min']
    df_norm = df_viz.copy()
    for f in feats:
        df_norm[f] = df_norm[f].rank(pct=True)
    
    df_avg = df_norm.groupby('dom_pos')[feats].mean().reset_index().melt('dom_pos')
    
    c_parallel = alt.Chart(df_avg).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('variable', title='Atributo'),
        y=alt.Y('value', title='Percentil (0-100%)', scale=alt.Scale(domain=[-0.1, 1.1])),
        color=alt.Color('dom_pos', title='Posición'),
        tooltip=['dom_pos', 'variable', alt.Tooltip('value', format='.2%')]
    ).properties(height=400).interactive()
    
    st.altair_chart(c_parallel, use_container_width=True)


# ==============================================================================
# PESTAÑA 3: PREDICCIÓN (SIMULADOR DE JUGADORES) - CORREGIDA
# ==============================================================================
with tab_pred:
    st.header("Simulador de Predicción")
    st.markdown("Busca un jugador real de la base de datos o selecciona un perfil simulado (VIP) para probar el modelo.")
    
    # 1. Buscador de Jugador
    if 'full_name' in df_viz.columns:
        player_list = sorted(df_viz['full_name'].unique())
        default_idx = player_list.index("LeBron James") if "LeBron James" in player_list else 0
        selected_player_name = st.selectbox("Seleccionar Jugador:", player_list, index=default_idx)
    else:
        st.error("El CSV no tiene la columna 'full_name'. Verifica la generación de datos.")
        selected_player_name = None
    
    if selected_player_name:
        # Obtener datos del jugador seleccionado
        player_row = df_viz[df_viz['full_name'] == selected_player_name].iloc[0]
        
        # 2. Ficha y Métricas
        st.subheader(f"Ficha: {selected_player_name}")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Altura", f"{player_row['height']:.0f}\"")
        col_b.metric("Peso", f"{player_row['bodyWeight']:.0f} lbs")
        col_c.metric("Posición Real", player_row['dom_pos'])
        col_d.metric("Edad", f"{player_row['age']:.1f}")

        st.divider()

        # --- INICIO: GRÁFICO DE BARRAS (ATRIBUTOS TÉCNICOS SOLAMENTE) ---
        st.subheader("Estadísticas de Perfil (Valores puros por 36 min)")

        # 1. Definir métricas y crear DataFrame
        # Quitamos: 'height', 'bodyWeight', 'age' para evitar redundancia con la ficha superior.
        display_metrics = {
            'Asistencias/36': 'ast36Min',
            'Rebotes/36': 'reb36Min',
            'Bloqueos/36': 'blk36Min',
            'Robos/36': 'stl36Min',
            'Pts. Dobles/36': 'pts_from_2_36Min',
            'Pts. Triples/36': 'pts_from_3_36Min'
        }
        
        bar_data = []
        for display_name, col_name in display_metrics.items():
            bar_data.append({
                'Atributo': display_name,
                'Valor': player_row[col_name],
                'Tipo': player_row['dom_pos']
            })
        df_bar = pd.DataFrame(bar_data)

        # 2. Crear el Gráfico de Barras
        c_bar = alt.Chart(df_bar).mark_bar().encode(
            # El valor va en el eje X
            x=alt.X('Valor:Q', title='Valor (No Normalizado)'),
            # El atributo va en el eje Y (ordenado por la lista)
            y=alt.Y('Atributo:N', sort=list(display_metrics.keys()), title=None),
            # Color por Posición Real
            color=alt.Color('Tipo', title='Posición Real'),
            # Tooltip con 2 decimales para precisión
            tooltip=['Atributo', alt.Tooltip('Valor', format=',.2f')]
        ).properties(
            title=f"Estadísticas Clave de {selected_player_name}",
            height=300
        )
        st.altair_chart(c_bar, use_container_width=True)
        # --- FIN: NUEVO GRÁFICO DE BARRAS ---

        # 3. Lógica de Predicción
        try:
            # Features requeridas por el pipeline (orden correcto)
            input_features = [
                'height', 'bodyWeight', 'birthdate',
                'points_acum', 'numMinutes_acum', 'reboundsTotal_acum', 'blocks_acum', 
                'assists_acum', 'steals_acum', 'threePointersMade_acum', 'fieldGoalsMade_acum'
            ]
            
            # --- NOTA: Debemos extraer la 'birthdate' como un objeto datetime ---
            input_data = player_row[input_features].to_frame().T
            input_data['birthdate'] = pd.to_datetime(input_data['birthdate']) 
            
            # PREDICCIÓN
            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]
            confidence = np.max(probs)
            
            # MOSTRAR RESULTADO
            st.divider()
            c_res1, c_res2 = st.columns([1, 2])
            
            with c_res1:
                st.subheader("El modelo dice:")
                if prediction == player_row['dom_pos']:
                    st.success(f"# ¡{prediction}!")
                    st.caption("Predicción Correcta ✅")
                else:
                    st.error(f"# ¡{prediction}!")
                    st.caption(f"Predicción Incorrecta ❌ (Era {player_row['dom_pos']})")
                
                st.progress(float(confidence), text=f"Confianza: {confidence:.1%}")
            
            # ANÁLISIS DE ERROR (GRÁFICO DE PERFIL / COORDENADAS PARALELAS)
            with c_res2:
                st.subheader("Análisis del Perfil")
                st.write("Comparamos el perfil del jugador (línea azul) con el promedio de su posición real y la predicha.")
                
                # 1. Features a comparar (normalizadas)
                radar_feats_viz = ['height', 'reb36Min', 'blk36Min', 'ast36Min', 'pts_from_3_36Min']
                
                # 2. Calcular percentiles globales para normalizar
                df_norm_radar = df_viz.copy()
                for f in radar_feats_viz:
                    df_norm_radar[f] = df_norm_radar[f].rank(pct=True)
                
                # 3. Datos del JUGADOR (normalizados)
                player_norm_data = df_norm_radar[df_norm_radar['full_name'] == selected_player_name].iloc[0]
                
                plot_data = []
                
                # Línea 1: El Jugador (Azul Oscuro o similar)
                for f in radar_feats_viz:
                    plot_data.append({'Feature': f, 'Valor': player_norm_data[f], 'Tipo': f'1. Jugador: {selected_player_name}'})
                
                # Línea 2: Promedio de la PREDICCIÓN
                avg_pred = df_norm_radar[df_norm_radar['dom_pos'] == prediction][radar_feats_viz].mean()
                for f in radar_feats_viz:
                    plot_data.append({'Feature': f, 'Valor': avg_pred[f], 'Tipo': f'2. Promedio {prediction} (Predicho)'})
                
                # Línea 3 (Solo si hubo error): Promedio de la REALIDAD
                if prediction != player_row['dom_pos']:
                    avg_real = df_norm_radar[df_norm_radar['dom_pos'] == player_row['dom_pos']][radar_feats_viz].mean()
                    for f in radar_feats_viz:
                        plot_data.append({'Feature': f, 'Valor': avg_real[f], 'Tipo': f'3. Promedio {player_row["dom_pos"]} (Real)'})
                
                df_radar_plot = pd.DataFrame(plot_data)
                
                # GRÁFICO DE LÍNEAS (PERFIL)
                c_profile = alt.Chart(df_radar_plot).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X('Feature', title='Atributo', sort=radar_feats_viz),
                    y=alt.Y('Valor', title='Percentil Relativo (0-1)', scale=alt.Scale(domain=[-0.1, 1.1])),
                    color=alt.Color('Tipo', title='Comparación', scale=alt.Scale(scheme='category10')),
                    tooltip=['Tipo', 'Feature', alt.Tooltip('Valor', format='.2%')]
                ).properties(height=300)
                
                st.altair_chart(c_profile, use_container_width=True)
                
                if prediction != player_row['dom_pos']:
                    st.warning(f"""
                    **¿Por qué el error?**
                    Observa el gráfico: La línea de **{selected_player_name}** sigue un patrón más parecido 
                    a la línea de **{prediction}** que a la de su posición real, **{player_row['dom_pos']}**.
                    """)
        
        except Exception as e:
            st.error(f"Error al predecir: {e}")