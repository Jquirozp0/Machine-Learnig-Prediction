# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# CONFIGURACIÓN INICIAL
# ------------------------------
st.set_page_config(page_title="Predicción de Rotación de Empleados", layout="wide")
st.title("🔍 Predicción de Rotación de Empleados (Attrition)")
st.write("Sube un archivo CSV con datos de empleados y visualiza el análisis completo de modelos de Machine Learning.")

# ------------------------------
# SUBIR DATASET
# ------------------------------
uploaded_file = st.file_uploader("📁 Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ------------------------------
    # PREPROCESAMIENTO
    # ------------------------------
    try:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

        features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'JobSatisfaction', 
                    'TotalWorkingYears', 'YearsAtCompany', 'OverTime']
        X = df[features]
        y = df['Attrition']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ------------------------------
        # ENTRENAMIENTO DE MODELOS
        # ------------------------------
        models = {
            "Regresión Logística": LogisticRegression(max_iter=1000),
            "Árbol de Decisión": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100)
        }

        st.subheader("📊 Reportes de Clasificación")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame(report).transpose())

        # ------------------------------
        # IMPORTANCIA DE VARIABLES
        # ------------------------------
        st.subheader("📌 Importancia de Variables (Random Forest)")
        rf = models["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        fig_imp, ax = plt.subplots()
        importances.sort_values().plot(kind='barh', ax=ax)
        ax.set_title("Importancia de Variables")
        st.pyplot(fig_imp)

        # ------------------------------
        # PRUEBA T DE SATISFACCIÓN
        # ------------------------------
        st.subheader("🧪 Prueba t: JobSatisfaction entre quienes renunciaron vs. no")
        job_sat_yes = df[df['Attrition'] == 1]['JobSatisfaction']
        job_sat_no = df[df['Attrition'] == 0]['JobSatisfaction']
        t_stat, p_val = ttest_ind(job_sat_yes, job_sat_no)
        st.write(f"**t = {t_stat:.2f}, p = {p_val:.4f}** {'✅ Significativo' if p_val < 0.05 else '❌ No significativo'}")

        # ------------------------------
        # BOXPLOT DE SATISFACCIÓN
        # ------------------------------
        st.subheader("📦 Boxplot: Satisfacción laboral vs. Rotación")
        fig_box, ax = plt.subplots()
        sns.boxplot(data=df, x='Attrition', y='JobSatisfaction', ax=ax)
        ax.set_xticklabels(["Se quedó", "Renunció"])
        ax.set_xlabel("Estado del empleado")
        ax.set_ylabel("Satisfacción laboral")
        st.pyplot(fig_box)

        # ------------------------------
        # CURVA ROC
        # ------------------------------
        st.subheader("📈 Curva ROC (Random Forest)")
        y_prob = rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel("Falsos Positivos")
        ax.set_ylabel("Verdaderos Positivos")
        ax.set_title("Curva ROC - Random Forest")
        ax.legend()
        st.pyplot(fig_roc)

        # ------------------------------
        # PREDICCIÓN PERSONALIZADA
        # ------------------------------
        st.subheader("🧮 Predicción Personalizada (Regresión Logística)")
        with st.form("form_pred"):
            col1, col2, col3 = st.columns(3)
            with col1:
                edad = st.slider("Edad", 18, 60, 35)
                distancia = st.slider("Distancia al trabajo (km)", 1, 50, 5)
            with col2:
                ingreso = st.slider("Ingreso mensual", 1000, 20000, 4500)
                satisfaccion = st.slider("Satisfacción laboral (1-4)", 1, 4, 3)
            with col3:
                años_totales = st.slider("Años de experiencia", 0, 40, 10)
                años_empresa = st.slider("Años en la empresa", 0, 20, 3)
                overtime = st.selectbox("¿Hace horas extra?", ["Sí", "No"])

            submit = st.form_submit_button("Predecir")

        if submit:
            overtime_bin = 1 if overtime == "Sí" else 0
            ejemplo = np.array([[edad, distancia, ingreso, satisfaccion, años_totales, años_empresa, overtime_bin]])
            prob = models["Regresión Logística"].predict_proba(ejemplo)[0][1]
            st.success(f"Probabilidad de que este empleado renuncie: **{prob:.2%}**")

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")

else:
    st.info("Por favor sube un archivo CSV para comenzar.")
