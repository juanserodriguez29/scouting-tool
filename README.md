# ⚽ Scouting Tool — Sudamérica & MLS

Herramienta de análisis y búsqueda de jugadores similares usando KMeans, construida con Streamlit.

---

## Estructura del proyecto

```
scouting_app/
├── app.py                              ← Aplicación Streamlit
├── KMeans_reemplazo_jugadores.ipynb   ← Notebook de scraping y entrenamiento
├── requirements.txt
├── data/
│   └── jugadores_2026.csv             ← Generado por el notebook
└── models/
    ├── scalers.pkl                    ← Generado por el notebook
    ├── kmeans_models.pkl              ← Generado por el notebook
    └── features_por_posicion.pkl      ← Generado por el notebook
```

---

## Cómo correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy en Streamlit Cloud (gratis, link compartible)

1. Sube todo este proyecto a un repositorio de GitHub (puede ser privado).
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **New app** y selecciona el repositorio.
4. En **Main file path** escribe: `app.py`
5. Haz clic en **Deploy**.

> ⚠️ **Importante:** El botón "Actualizar datos" en la app corre el notebook completo.
> En Streamlit Cloud esto requiere que Chrome/Chromium esté disponible en el servidor.
> Si hay problemas con el scraping en la nube, lo más confiable es correr el notebook
> localmente, hacer commit de `data/jugadores_2026.csv` y los archivos `.pkl` al repo,
> y dejar que la app solo lea esos archivos.

---

## Flujos de la app

### 🏆 Ranking de Jugadores
- Filtra por posición, liga(s), minutos mínimos jugados
- Ordena por cualquier métrica de la posición seleccionada
- Descarga la tabla en CSV

### 🔍 Buscar Similar
- Selecciona posición y jugador de referencia
- La app calcula distancia euclidiana en el espacio de features escaladas
- Muestra los N jugadores más similares con tabla comparativa
- Descarga la comparativa en CSV
