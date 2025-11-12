# Dashboard-taller-final.py
"""
Dashboard Estudiantil - Final (sin experimental_data_editor)
- Guarda en la misma carpeta que base_dashboard.csv (opcional)
- Ejecutar:
    pip install streamlit pandas numpy plotly openpyxl
    streamlit run Dashboard-taller-final.py
"""

import os
from io import BytesIO
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Config
st.set_page_config(page_title="Dashboard Estudiantil", layout="wide")

# ---------------- Helpers ----------------
def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    cols_low = {col.lower().strip(): col for col in df.columns}
    for c in candidates:
        if c.lower().strip() in cols_low:
            return cols_low[c.lower().strip()]
    return None

def to_numeric_clean(series):
    s = series.astype(str).str.strip()
    # if commas exist and no dots, treat commas as decimal separators
    has_comma = s.str.contains(",").any()
    has_dot = s.str.contains("\\.").any()
    if has_comma and not has_dot:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def parse_dates_aggressive(series):
    d = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if d.notna().sum() > 0:
        return d
    return pd.to_datetime(series, dayfirst=False, errors="coerce")

def calculate_age_from_dob(series):
    today = pd.to_datetime(datetime.today()).normalize()
    dob = pd.to_datetime(series, errors="coerce")
    years = today.year - dob.dt.year
    had_bday = ((today.month > dob.dt.month) | ((today.month == dob.dt.month) & (today.day >= dob.dt.day)))
    years = years - (~had_bday).astype(int)
    return years.where(~dob.isna())

def classify_imc(v):
    if pd.isna(v): return pd.NA
    v = float(v)
    if v < 18.5: return "Bajo peso"
    if v <= 24.9: return "Adecuado"
    if v <= 29.9: return "Sobrepeso"
    if v <= 34.9: return "Obesidad grado 1"
    if v <= 39.9: return "Obesidad grado 2"
    return "Obesidad grado 3"

def value_counts_table(series):
    vc = series.value_counts(dropna=False).reset_index()
    vc.columns = ['category', 'count']
    vc['category'] = vc['category'].astype(str)
    return vc

# Optional trendline dependency
try:
    import statsmodels.api  # noqa: F401
    STATSMODELS = True
except Exception:
    STATSMODELS = False

# ---------------- Sidebar & Load ----------------
st.sidebar.header("Instrucciones")
st.sidebar.write("Sube un archivo .csv con tus datos o deja `base_dashboard.csv` en la carpeta del script.")
uploaded = st.sidebar.file_uploader("Subir archivo (solo .csv)", type=['csv'])
use_sample = st.sidebar.button("Cargar muestra de ejemplo (40 filas)")
group_input = st.sidebar.text_input("Nombre del grupo / etiqueta", value="Grupo xxx (001,050, 051)")
show_raw = st.sidebar.checkbox("Mostrar archivo original completo", value=False)

# Load data
df = None
if use_sample and not uploaded:
    n = 40
    np.random.seed(42)
    df = pd.DataFrame({
        'Nombre': [f'Estudiante {i}' for i in range(1, n+1)],
        'Fecha de nacimiento': pd.date_range('2006-01-01', periods=n, freq='90D'),
        'Estatura': np.round(np.random.normal(1.65, 0.08, n), 2),
        'Peso': np.round(np.random.normal(60, 8, n), 1),
        'Tipo de sangre': np.random.choice(['A+','A-','B+','B-','O+','O-','AB+'], n),
        'Color de Cabello': np.random.choice(['Negro','Castaño','Rubio','Pelirrojo'], n),
        'Barrio': np.random.choice([f'Barrio {i}' for i in range(1,21)], n),
        'Talla Zapato': np.random.choice(range(34,45), n)
    })
elif uploaded is not None:
    try:
        raw = uploaded.read()
        df = pd.read_csv(BytesIO(raw))
    except Exception as e:
        st.error(f"Error leyendo el CSV subido: {e}")
        st.stop()
else:
    local_name = "base_dashboard.csv"
    cwd = os.getcwd()
    local_path = os.path.join(cwd, local_name)
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            st.sidebar.success(f"Cargado archivo local: {local_name}")
        except Exception as e:
            st.error(f"No se pudo leer {local_name}: {e}")
            st.stop()
    else:
        st.info("Sube un CSV o pulsa 'Cargar muestra de ejemplo' en la barra lateral.")
        st.stop()

# Quick preview
st.header("Archivo cargado")
source = "uploader" if uploaded is not None else ("muestra" if use_sample else local_name)
st.write(f"Origen: {source}")
if show_raw:
    st.subheader("Archivo original (completo)")
    st.dataframe(df)
else:
    st.subheader("Preview (primeras 10 filas)")
    st.dataframe(df.head(10))
st.write("Columnas detectadas:", list(df.columns))

# ---------------- Normalize & Add Columns ----------------
st.info("Normalizando datos y agregando columnas: Edad, Peso (kg), IMC, Clasificación IMC, Estatura (cm)")

# detect candidates
col_est = detect_column(df, ['Estatura (m)','Estatura (cm)','Estatura','estatura','Height','Altura','estatura_m'])
col_peso = detect_column(df, ['Peso (kg)','Peso','peso','Weight'])
col_dob  = detect_column(df, ['Fecha de nacimiento','FechaNacimiento','Fecha_Nacimiento','DOB','dob','birthdate'])
col_age  = detect_column(df, ['Edad','edad','Age'])
col_rh   = detect_column(df, ['Tipo de Sangre','Tipo_de_Sangre','RH','Blood Type','blood_type'])
col_hair = detect_column(df, ['Color de Cabello','ColorCabello','Cabello','Hair Color','hair_color'])
col_barrio = detect_column(df, ['Barrio','Barrio de Residencia','Neighborhood','neighborhood'])
col_shoe = detect_column(df, ['Talla Zapato','Talla_Zapato','Shoe Size','shoe_size'])
col_name = detect_column(df, ['Nombre','name','Name'])

# Estatura normalization (prioritize existing)
if 'Estatura (m)' in df.columns and 'Estatura (cm)' in df.columns:
    df['Estatura (m)'] = to_numeric_clean(df['Estatura (m)'])
    df['Estatura (cm)'] = to_numeric_clean(df['Estatura (cm)']).round(0).astype('Int64', errors='ignore')
elif 'Estatura (cm)' in df.columns:
    df['Estatura (cm)'] = to_numeric_clean(df['Estatura (cm)']).round(0).astype('Int64', errors='ignore')
    df['Estatura (m)'] = (to_numeric_clean(df['Estatura (cm)']) / 100).round(2)
elif 'Estatura (m)' in df.columns:
    df['Estatura (m)'] = to_numeric_clean(df['Estatura (m)']).round(2)
    df['Estatura (cm)'] = (df['Estatura (m)'] * 100).round(0).astype('Int64', errors='ignore')
else:
    if col_est:
        s = to_numeric_clean(df[col_est])
        if s.notna().any() and s.max() <= 3.0:
            df['Estatura (m)'] = s.round(2)
            df['Estatura (cm)'] = (df['Estatura (m)'] * 100).round(0).astype('Int64', errors='ignore')
        else:
            df['Estatura (cm)'] = s.round(0).astype('Int64', errors='ignore')
            df['Estatura (m)'] = (to_numeric_clean(df['Estatura (cm)']) / 100).round(2)
    else:
        df['Estatura (m)'] = pd.NA
        df['Estatura (cm)'] = pd.NA

# Peso
if 'Peso (kg)' in df.columns:
    df['Peso (kg)'] = to_numeric_clean(df['Peso (kg)']).round(1)
elif col_peso:
    df['Peso (kg)'] = to_numeric_clean(df[col_peso]).round(1)
else:
    df['Peso (kg)'] = pd.NA

# Edad
if 'Edad' in df.columns and col_age:
    df['Edad'] = to_numeric_clean(df[col_age]).astype('Int64', errors='ignore')
elif col_dob:
    parsed = parse_dates_aggressive(df[col_dob])
    df['Edad'] = calculate_age_from_dob(parsed).astype('Int64', errors='ignore')
else:
    df['Edad'] = pd.NA

# IMC
if 'IMC' in df.columns:
    df['IMC'] = to_numeric_clean(df['IMC']).round(1)
else:
    est_m = to_numeric_clean(df.get('Estatura (m)', pd.Series([pd.NA]*len(df))))
    peso_n = to_numeric_clean(df.get('Peso (kg)', pd.Series([pd.NA]*len(df))))
    with np.errstate(divide='ignore', invalid='ignore'):
        df['IMC'] = (peso_n / (est_m ** 2)).round(1)

# Clasificación IMC
if 'Clasificación IMC' not in df.columns:
    df['Clasificación IMC'] = df['IMC'].apply(lambda x: classify_imc(x) if not pd.isna(x) else pd.NA)

# Move new cols to the end
base_cols = [c for c in df.columns if c not in ['Estatura (m)','Estatura (cm)','Edad','Peso (kg)','IMC','Clasificación IMC']]
final_cols = base_cols + [c for c in ['Estatura (m)','Estatura (cm)','Edad','Peso (kg)','IMC','Clasificación IMC'] if c in df.columns]
df = df.loc[:, final_cols]

st.success("Columnas creadas/normalizadas.")
st.subheader("Preview dataset procesado (primeras 10 filas)")
st.dataframe(df.head(10))

# Title after showing file
st.markdown("---")
st.title(f"Dashboard Estudiantil – {group_input}")

# ---------------- Filters ----------------
st.sidebar.header("Filtros")
if col_rh and col_rh in df.columns:
    rh_options = sorted(df[col_rh].dropna().unique())
    rh_sel = st.sidebar.multiselect("Tipo de Sangre (RH)", options=rh_options, default=rh_options)
else:
    rh_sel = []

if col_hair and col_hair in df.columns:
    hair_options = sorted(df[col_hair].dropna().unique())
    hair_sel = st.sidebar.multiselect("Color de Cabello", options=hair_options, default=hair_options)
else:
    hair_sel = []

if col_barrio and col_barrio in df.columns:
    barrio_options = sorted(df[col_barrio].dropna().unique())
    barrio_sel = st.sidebar.multiselect("Barrio de Residencia", options=barrio_options, default=barrio_options)
else:
    barrio_sel = []

st.sidebar.markdown("---")
min_age = int(df['Edad'].min()) if 'Edad' in df.columns and df['Edad'].notna().any() else 0
max_age = int(df['Edad'].max()) if 'Edad' in df.columns and df['Edad'].notna().any() else 100
age_range = st.sidebar.slider("Rango de Edad", min_value=min_age, max_value=max_age, value=(min_age, max_age))

min_est = int(df['Estatura (cm)'].min()) if 'Estatura (cm)' in df.columns and df['Estatura (cm)'].notna().any() else 100
max_est = int(df['Estatura (cm)'].max()) if 'Estatura (cm)' in df.columns and df['Estatura (cm)'].notna().any() else 220
est_range = st.sidebar.slider("Rango de Estatura (cm)", min_value=min_est, max_value=max_est, value=(min_est, max_est))

# Apply filters
df_f = df.copy()
if rh_sel and col_rh:
    df_f = df_f[df_f[col_rh].isin(rh_sel)]
if hair_sel and col_hair:
    df_f = df_f[df_f[col_hair].isin(hair_sel)]
if barrio_sel and col_barrio:
    df_f = df_f[df_f[col_barrio].isin(barrio_sel)]

if 'Edad' in df_f.columns:
    df_f = df_f[(df_f['Edad'] >= age_range[0]) & (df_f['Edad'] <= age_range[1])]
if 'Estatura (cm)' in df_f.columns:
    df_f = df_f[(df_f['Estatura (cm)'] >= est_range[0]) & (df_f['Estatura (cm)'] <= est_range[1])]

# ---------------- KPIs ----------------
st.markdown("### Resumen rápido")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Estudiantes", int(df_f.shape[0]))
k2.metric("Edad Promedio", f"{df_f['Edad'].mean():.1f}" if 'Edad' in df_f.columns and df_f['Edad'].notna().any() else "N/A")
k3.metric("Estatura Promedio (cm)", f"{df_f['Estatura (cm)'].mean():.1f}" if 'Estatura (cm)' in df_f.columns and df_f['Estatura (cm)'].notna().any() else "N/A")
k4.metric("Peso Promedio (kg)", f"{df_f['Peso (kg)'].mean():.1f}" if 'Peso (kg)' in df_f.columns and df_f['Peso (kg)'].notna().any() else "N/A")
k5.metric("IMC Promedio", f"{df_f['IMC'].mean():.1f}" if 'IMC' in df_f.columns and df_f['IMC'].notna().any() else "N/A")

# ---------------- Charts ----------------
st.markdown("---")
st.markdown("## Gráficos")

# Row 1
r1c1, r1c2 = st.columns([2,1])
with r1c1:
    st.subheader("Distribución por Edad (barras)")
    if 'Edad' in df_f.columns and df_f['Edad'].notna().any():
        fig_age = px.histogram(df_f, x='Edad', nbins=10, title="Distribución por Edad")
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.info("No hay datos de Edad para graficar.")
with r1c2:
    st.subheader("Distribución por Tipo de Sangre (torta)")
    if col_rh:
        fig_rh = px.pie(df_f, names=col_rh, title="Tipo de Sangre")
        st.plotly_chart(fig_rh, use_container_width=True)
    else:
        st.info("No hay columna Tipo de Sangre.")

# Row 2
r2c1, r2c2 = st.columns([2,1])
with r2c1:
    st.subheader("Relación Estatura vs Peso (scatter)")
    if 'Estatura (cm)' in df_f.columns and 'Peso (kg)' in df_f.columns and df_f['Estatura (cm)'].notna().any() and df_f['Peso (kg)'].notna().any():
        if STATSMODELS:
            fig_sc = px.scatter(df_f, x='Estatura (cm)', y='Peso (kg)', hover_data=[col_name] if col_name else None, trendline='ols', labels={'Estatura (cm)':'Estatura (cm)','Peso (kg)':'Peso (kg)'})
        else:
            fig_sc = px.scatter(df_f, x='Estatura (cm)', y='Peso (kg)', hover_data=[col_name] if col_name else None, labels={'Estatura (cm)':'Estatura (cm)','Peso (kg)':'Peso (kg)'})
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Faltan Estatura o Peso para scatter.")
with r2c2:
    st.subheader("Distribución por Color de Cabello (barras)")
    if col_hair:
        vc_hair = value_counts_table(df_f[col_hair])
        fig_hair = px.bar(vc_hair, x='category', y='count', labels={'category': col_hair, 'count': 'Cantidad'}, title='Color de Cabello')
        st.plotly_chart(fig_hair, use_container_width=True)
    else:
        st.info("No hay columna Color de Cabello.")

# Row 3
r3c1, r3c2 = st.columns([2,1])
with r3c1:
    st.subheader("Distribución de Tallas de Zapato (línea)")
    if col_shoe:
        vc_shoe = value_counts_table(df_f[col_shoe])
        fig_shoe = px.line(vc_shoe, x='category', y='count', labels={'category': 'Talla', 'count': 'Cantidad'}, title='Tallas de Zapato')
        st.plotly_chart(fig_shoe, use_container_width=True)
    else:
        st.info("No hay columna Talla Zapato.")
with r3c2:
    st.subheader("Top 10 Barrios de residencia")
    if col_barrio:
        vc_barrio = value_counts_table(df_f[col_barrio]).head(10)
        fig_barrio = px.bar(vc_barrio, x='category', y='count', labels={'category': 'Barrio', 'count': 'Cantidad'}, title='Top 10 Barrios')
        st.plotly_chart(fig_barrio, use_container_width=True)
    else:
        st.info("No hay columna Barrio.")

# ---------------- Top5 & downloads ----------------
st.markdown("---")
st.subheader("Top 5 - Exportar")
left, right = st.columns(2)
with left:
    if 'Estatura (cm)' in df_f.columns:
        top5_h = df_f.sort_values('Estatura (cm)', ascending=False).head(5)
        st.write("Top 5 Mayor Estatura")
        st.dataframe(top5_h)
        st.download_button("Descargar Top 5 Mayor Estatura (CSV)", data=top5_h.to_csv(index=False).encode('utf-8'), file_name='top5_estatura.csv', mime='text/csv')
    else:
        st.info("No hay Estatura (cm) para Top5.")
with right:
    if 'Peso (kg)' in df_f.columns:
        top5_w = df_f.sort_values('Peso (kg)', ascending=False).head(5)
        st.write("Top 5 Mayor Peso")
        st.dataframe(top5_w)
        st.download_button("Descargar Top 5 Mayor Peso (CSV)", data=top5_w.to_csv(index=False).encode('utf-8'), file_name='top5_peso.csv', mime='text/csv')
    else:
        st.info("No hay Peso (kg) para Top5.")

# ---------------- Summary statistics ----------------
st.markdown("---")
st.subheader("Resumen Estadístico")
cols_stat = [c for c in ['Estatura (cm)', 'Peso (kg)', 'IMC'] if c in df_f.columns]
if cols_stat:
    st.dataframe(df_f[cols_stat].describe().T)
else:
    st.info("No hay columnas numéricas suficientes para resumen estadístico.")

# ---------------- Download processed dataset ----------------
st.markdown("---")
st.subheader("Descargar dataset procesado")
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("Descargar dataset completo (CSV)", data=csv_bytes, file_name='dataset_procesado.csv', mime='text/csv')

# IMC table
st.markdown("---")
st.subheader("Tabla referencia: Clasificación IMC (OMS)")
st.table(pd.DataFrame([
    {"IMC": "<18.5", "Clasificación": "Bajo peso"},
    {"IMC": "18.5 - 24.9", "Clasificación": "Adecuado"},
    {"IMC": "25.0 - 29.9", "Clasificación": "Sobrepeso"},
    {"IMC": "30.0 - 34.9", "Clasificación": "Obesidad grado 1"},
    {"IMC": "35.0 - 39.9", "Clasificación": "Obesidad grado 2"},
    {"IMC": ">=40.0", "Clasificación": "Obesidad grado 3"},
]))

if not STATSMODELS:
    st.warning("Si quieres la línea de tendencia en el scatter instala 'statsmodels' (pip install statsmodels). Actualmente trendline está desactivado.")

st.caption("Desarrollado con Streamlit — Dashboard Estudiantil.")
