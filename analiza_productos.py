# analiza_productos_bidireccional.py
# Autor: Salvador + GPT-5
# Descripción:
# Compara productos de Odoo con productos del cliente usando TF-IDF.
# Permite búsqueda bidireccional y genera archivo de salida con coincidencias.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========================
# CONFIGURACIÓN BÁSICA
# ========================
CATALOGO_FILE = "product_template.csv"
FORMATO_FILE = "FORMATO PEDIDO BASES 2024.csv"
OUTPUT_FILE = "productos_relacionados.csv"
UMBRAL_COINCIDENCIA = 0.070
# Dirección de búsqueda: "catalogo_a_cliente" o "cliente_a_catalogo"
DIRECCION_BUSQUEDA = "catalogo_a_cliente"

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
print("Calculando TF-IDF y similitudes...")
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

if DIRECCION_BUSQUEDA == "catalogo_a_cliente":
    tfidf_catalogo = vectorizer.fit_transform(catalogo["Name"])
    tfidf_formato = vectorizer.transform(formato["DESCRIPCION"])
    sim_matrix = cosine_similarity(tfidf_catalogo, tfidf_formato)
    df_base = catalogo.copy()
    df_busqueda = formato
    col_match_name = "Descripcion_cliente"
    col_match_key = "Clave_cliente"
    col_match_source = "DESCRIPCION"
    col_match_key_source = "ARTICULO"
elif DIRECCION_BUSQUEDA == "cliente_a_catalogo":
    tfidf_formato = vectorizer.fit_transform(formato["DESCRIPCION"])
    tfidf_catalogo = vectorizer.transform(catalogo["Name"])
    sim_matrix = cosine_similarity(tfidf_formato, tfidf_catalogo)
    df_base = formato.copy()
    df_busqueda = catalogo
    col_match_name = "Producto_catalogo"
    col_match_key = None  # Opcional si no quieres clave
else:
    raise ValueError("DIRECCION_BUSQUEDA debe ser 'catalogo_a_cliente' o 'cliente_a_catalogo'")

# ========================
# SELECCIONAR MEJOR COINCIDENCIA
# ========================
mejores_indices = np.argmax(sim_matrix, axis=1)
mejores_scores = np.max(sim_matrix, axis=1)

# ========================
# CREAR SALIDA
# ========================
salida = df_base.copy()
if col_match_key:
    salida[col_match_name] = [df_busqueda.iloc[i][col_match_source] for i in mejores_indices]
    salida[col_match_key] = [df_busqueda.iloc[i][col_match_key_source] for i in mejores_indices]
else:
    salida[col_match_name] = [df_busqueda.iloc[i]["Name"] for i in mejores_indices]

salida["Similitud"] = np.round(mejores_scores, 3)

# Marcamos coincidencias débiles
salida.loc[salida["Similitud"] < UMBRAL_COINCIDENCIA, col_match_name] = "SIN_COINCIDENCIA"
if col_match_key:
    salida.loc[salida["Similitud"] < UMBRAL_COINCIDENCIA, col_match_key] = "SIN_COINCIDENCIA"

# ========================
# GUARDAR RESULTADO
# ========================
salida.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n✅ Archivo generado: {OUTPUT_FILE}")

# ========================
# PREVISUALIZACIÓN
# ========================
print("\nEjemplo de coincidencias:")
cols_preview = [col for col in [col_match_name, col_match_key, "Similitud"] if col]
preview = salida[cols_preview].head(10)
print(preview.to_string(index=False))

# ========================
# RESUMEN
# ========================
total = len(salida)
con_coinc = (salida[col_match_name] != "SIN_COINCIDENCIA").sum()
sin_coinc = total - con_coinc
porcentaje = (con_coinc / total) * 100 if total > 0 else 0

print("\n📊 --- RESUMEN DEL ANÁLISIS ---")
print(f"Total de registros procesados: {total}")
print(f"Con coincidencia ≥ {UMBRAL_COINCIDENCIA}: {con_coinc}")
print(f"Sin coincidencia: {sin_coinc}")
print(f"Porcentaje de coincidencias: {porcentaje:.2f}%")
print("------------------------------")
