# analiza_productos_bidireccional_completo.py
# Autor: Salvador + GPT-5
# Descripción:
# Compara productos de Odoo con productos del cliente usando TF-IDF.
# Genera coincidencias bidireccionales en un solo archivo de salida.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========================
# CONFIGURACIÓN
# ========================
CATALOGO_FILE = "product_template.csv"
FORMATO_FILE = "FORMATO PEDIDO BASES 2024.csv"
OUTPUT_FILE = "productos_relacionados_bidireccional.csv"
UMBRAL_COINCIDENCIA = 0.75

# ========================
# CARGA DE ARCHIVOS
# ========================
print("Cargando archivos...")
catalogo = pd.read_csv(CATALOGO_FILE)
formato = pd.read_csv(FORMATO_FILE)

# Validar columnas necesarias
if "Name" not in catalogo.columns:
    raise ValueError("No se encontró la columna 'Name' en el archivo product_template.csv")
if not any(c in formato.columns for c in ["DESCRIPCION", "ARTICULO"]):
    raise ValueError("No se encontraron las columnas 'DESCRIPCION' o 'ARTICULO' en el formato del cliente")

# ========================
# PREPROCESAMIENTO
# ========================
catalogo["Name"] = catalogo["Name"].astype(str).str.lower()
formato["DESCRIPCION"] = formato["DESCRIPCION"].astype(str).str.lower()
formato["ARTICULO"] = formato["ARTICULO"].astype(str).str.lower()

# Eliminamos duplicados y nulos
formato = formato.dropna(subset=["DESCRIPCION"]).drop_duplicates(subset=["DESCRIPCION"])

# ========================
# TF-IDF
# ========================
print("Calculando TF-IDF...")
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

# Catalogo → Cliente
tfidf_catalogo = vectorizer.fit_transform(catalogo["Name"])
tfidf_formato = vectorizer.transform(formato["DESCRIPCION"])
sim_cat_cliente = cosine_similarity(tfidf_catalogo, tfidf_formato)

# Cliente → Catalogo
tfidf_formato2 = vectorizer.fit_transform(formato["DESCRIPCION"])
tfidf_catalogo2 = vectorizer.transform(catalogo["Name"])
sim_cliente_cat = cosine_similarity(tfidf_formato2, tfidf_catalogo2)

# ========================
# Mejor coincidencia Catalogo → Cliente
# ========================
indices_cat_cliente = np.argmax(sim_cat_cliente, axis=1)
scores_cat_cliente = np.max(sim_cat_cliente, axis=1)
salida_cat_cliente = catalogo.copy()
salida_cat_cliente["Descripcion_cliente"] = [formato.iloc[i]["DESCRIPCION"] for i in indices_cat_cliente]
salida_cat_cliente["Clave_cliente"] = [formato.iloc[i]["ARTICULO"] for i in indices_cat_cliente]
salida_cat_cliente["Similitud_cat_cliente"] = np.round(scores_cat_cliente, 3)
salida_cat_cliente.loc[salida_cat_cliente["Similitud_cat_cliente"] < UMBRAL_COINCIDENCIA, ["Descripcion_cliente","Clave_cliente"]] = "SIN_COINCIDENCIA"

# ========================
# Mejor coincidencia Cliente → Catalogo
# ========================
indices_cliente_cat = np.argmax(sim_cliente_cat, axis=1)
scores_cliente_cat = np.max(sim_cliente_cat, axis=1)
salida_cliente_cat = formato.copy()
salida_cliente_cat["Producto_catalogo"] = [catalogo.iloc[i]["Name"] for i in indices_cliente_cat]
salida_cliente_cat["Similitud_cliente_cat"] = np.round(scores_cliente_cat, 3)
salida_cliente_cat.loc[salida_cliente_cat["Similitud_cliente_cat"] < UMBRAL_COINCIDENCIA, ["Producto_catalogo"]] = "SIN_COINCIDENCIA"

# ========================
# COMBINAR SALIDAS
# ========================
# Guardamos ambos resultados en un solo Excel/CSV con dos pestañas
with pd.ExcelWriter(OUTPUT_FILE.replace(".csv",".xlsx"), engine="xlsxwriter") as writer:
    salida_cat_cliente.to_excel(writer, sheet_name="Catalogo_a_Cliente", index=False)
    salida_cliente_cat.to_excel(writer, sheet_name="Cliente_a_Catalogo", index=False)

print(f"\n✅ Archivo Excel generado: {OUTPUT_FILE.replace('.csv','.xlsx')}")

# ========================
# PREVISUALIZACIÓN
# ========================
print("\nEjemplo coincidencias Catalogo → Cliente:")
print(salida_cat_cliente[["Name","Descripcion_cliente","Clave_cliente","Similitud_cat_cliente"]].head(5).to_string(index=False))

print("\nEjemplo coincidencias Cliente → Catalogo:")
print(salida_cliente_cat[["DESCRIPCION","Producto_catalogo","Similitud_cliente_cat"]].head(5).to_string(index=False))

# ========================
# RESUMEN
# ========================
def resumen(df, col_match, score_col, umbral):
    total = len(df)
    con_coinc = (df[col_match] != "SIN_COINCIDENCIA").sum()
    sin_coinc = total - con_coinc
    pct = (con_coinc / total * 100) if total > 0 else 0
    return total, con_coinc, sin_coinc, pct

tot, con, sinc, pct = resumen(salida_cat_cliente, "Descripcion_cliente", "Similitud_cat_cliente", UMBRAL_COINCIDENCIA)
print(f"\n📊 Catalogo → Cliente: {con}/{tot} coincidencias ≥ {UMBRAL_COINCIDENCIA} ({pct:.2f}%)")
tot, con, sinc, pct = resumen(salida_cliente_cat, "Producto_catalogo", "Similitud_cliente_cat", UMBRAL_COINCIDENCIA)
print(f"📊 Cliente → Catalogo: {con}/{tot} coincidencias ≥ {UMBRAL_COINCIDENCIA} ({pct:.2f}%)")
