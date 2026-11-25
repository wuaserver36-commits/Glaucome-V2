import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from grille_24_2 import COLONNES_TD, POINTS_OD, TACHE_AVEUGLE_OD, og_vers_od
from modele_glaucome import charger, creer_xgb, nombre, regler_hyperparametres

# Configuration générale de la page
st.set_page_config(
    page_title="Glaucome 24-2 | Aide à l'analyse",
    page_icon="👁️",
    layout="wide",
)

DOSSIER_RESULTATS = Path("resultats_test")
CHEMIN_MODELE = DOSSIER_RESULTATS / "modele_xgboost.json"
CHEMIN_JSON = DOSSIER_RESULTATS / "resultats.json"
CHEMIN_HISTORIQUE = Path("historique_analyses.csv")


# ---------------------------------------------------------------- Fonctions utilitaires
@st.cache_resource
def charger_modele_et_seuils():
    """Charge le modèle XGBoost entraîné et les seuils enregistrés."""
    if not CHEMIN_MODELE.exists():
        return None, {"youden": 0.50, "specificite_90": 0.74, "specificite_95": 0.93}
    modele = xgb.XGBClassifier()
    modele.load_model(CHEMIN_MODELE)
    seuils = {"youden": 0.50, "specificite_90": 0.74, "specificite_95": 0.93}
    if CHEMIN_JSON.exists():
        try:
            data = json.loads(CHEMIN_JSON.read_text(encoding="utf-8"))
            seuils.update(data.get("seuils", {}))
        except Exception:
            pass
    return modele, seuils


def interpreter_risque(proba, seuils):
    """Traduit une probabilité technique en message clair et compréhensible."""
    seuil_modere = seuils.get("youden", 0.50)
    seuil_eleve = seuils.get("specificite_90", 0.74)

    if proba >= seuil_eleve:
        return (
            "Risque ÉLEVÉ d'atteinte glaucomateuse",
            "error",
            "Le profil des 52 points du champ visuel présente des anomalies marquées compatibles avec un glaucome débutant. Un examen ophtalmologique approfondi (OCT, fond d'œil, pression intraoculaire) est recommandé.",
        )
    elif proba >= seuil_modere:
        return (
            "Risque MODÉRÉ — À surveiller",
            "warning",
            "Certaines zones du champ visuel montrent une baisse de sensibilité atypique. Une confrontation avec les données cliniques et un contrôle rapproché sont conseillés.",
        )
    else:
        return (
            "Risque FAIBLE — Champ visuel rassurant",
            "success",
            "Les valeurs de déviation totale sont globalement dans les limites attendues. Aucune signature nette de glaucome détectée par le modèle.",
        )


def dessiner_carte_cv(valeurs_od, contribs=None, titre="Carte 24-2 (orientation œil droit)"):
    """Affiche la grille 24-2 (sensibilité TD en dB et zones les plus influentes)."""
    pts = np.array(POINTS_OD)
    fig, axes = plt.subplots(1, 2 if contribs is not None else 1, figsize=(11 if contribs is not None else 6, 5))
    if contribs is None:
        axes = [axes]

    # Carte 1 : Déviation totale (dB)
    ax0 = axes[0]
    sc0 = ax0.scatter(
        pts[:, 0], pts[:, 1], c=valeurs_od, cmap="RdYlGn", vmin=-15, vmax=3,
        s=650, marker="s", edgecolors="grey"
    )
    for (x, y), v in zip(POINTS_OD, valeurs_od):
        ax0.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=8, fontweight="bold")
    for x, y in TACHE_AVEUGLE_OD:
        ax0.scatter(x, y, s=650, marker="s", color="black")
    ax0.axhline(0, color="grey", lw=0.5)
    ax0.axvline(0, color="grey", lw=0.5)
    ax0.set_xlim(-32, 32)
    ax0.set_ylim(-26, 26)
    ax0.set_aspect("equal")
    ax0.set_xlabel("Nasal  <-  degrés  ->  Temporal")
    ax0.set_title("Déviation totale (dB)")
    fig.colorbar(sc0, ax=ax0, shrink=0.75, label="dB")

    # Carte 2 : Zones ayant le plus pesé dans la décision
    if contribs is not None:
        ax1 = axes[1]
        sc1 = ax1.scatter(
            pts[:, 0], pts[:, 1], c=contribs, cmap="Reds",
            s=650, marker="s", edgecolors="grey"
        )
        for (x, y), v in zip(POINTS_OD, contribs):
            if v > 0.02:
                ax1.text(x, y, f"+{v:.2f}", ha="center", va="center", fontsize=7)
        for x, y in TACHE_AVEUGLE_OD:
            ax1.scatter(x, y, s=650, marker="s", color="black")
        ax1.axhline(0, color="grey", lw=0.5)
        ax1.axvline(0, color="grey", lw=0.5)
        ax1.set_xlim(-32, 32)
        ax1.set_ylim(-26, 26)
        ax1.set_aspect("equal")
        ax1.set_xlabel("Nasal  <-  degrés  ->  Temporal")
        ax1.set_title("Zones suspectes détectées (impact)")
        fig.colorbar(sc1, ax=ax1, shrink=0.75, label="Impact")

    fig.suptitle(titre, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def enregistrer_historique(lignes_histo):
    """Ajoute les analyses effectuées dans un fichier CSV local."""
    df_new = pd.DataFrame(lignes_histo)
    if CHEMIN_HISTORIQUE.exists():
        df_old = pd.read_csv(CHEMIN_HISTORIQUE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CHEMIN_HISTORIQUE, index=False)


# ---------------------------------------------------------------- Navigation latérale
st.sidebar.title("👁️ Glaucome 24-2")
st.sidebar.caption("Prototype d'aide à l'analyse du champ visuel")

menu = st.sidebar.radio(
    "Navigation",
    [
        "1. Accueil",
        "2. Importer un examen",
        "3. Analyser",
        "4. Résultats compréhensibles",
        "5. Historique & Rapports",
    ],
)

modele, seuils = charger_modele_et_seuils()

st.sidebar.divider()
if modele is not None:
    st.sidebar.success("Modèle IA prêt (XGBoost)")
else:
    st.sidebar.warning("Aucun modèle entraîné trouvé")

st.sidebar.info(
    "⚠️ **Outil de recherche / prototype** : ne remplace pas un diagnostic médical."
)


# ---------------------------------------------------------------- 1. ACCUEIL
if menu == "1. Accueil":
    st.title("Aide à la détection précoce du glaucome (Champ Visuel 24-2)")
    st.markdown(
        """
        Cette application permet d'analyser les **52 points de déviation totale** d'un champ visuel
        Humphrey 24-2 pour détecter des signes précoces de glaucome, avec une restitution claire
        pour le praticien et le patient.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Données analysées", "52 points (24-2)", "OD et OG supportés")
    with col2:
        st.metric("État du modèle", "Prêt" if modele else "À entraîner", "XGBoost 24-2")
    with col3:
        nb_histo = len(pd.read_csv(CHEMIN_HISTORIQUE)) if CHEMIN_HISTORIQUE.exists() else 0
        st.metric("Examens analysés", f"{nb_histo}", "Enregistrés localement")

    st.subheader("Comment utiliser l'application ?")
    st.markdown(
        """
        1. **Importer un examen** : chargez un fichier `.csv` (comme `donnees_test.csv`) ou saisissez un examen. *(L'import direct de rapports PDF 24-2 sera ajouté à l'étape suivante.)*
        2. **Analyser** : lancez l'évaluation automatique sur les yeux importés.
        3. **Résultats compréhensibles** : consultez le niveau de risque en langage clair, la carte visuelle du champ visuel et les détails médicaux.
        4. **Historique & Rapports** : retrouvez et exportez vos bilans précédents.
        """
    )

    if modele is None:
        st.warning("Le fichier `resultats_test/modele_xgboost.json` n'a pas été trouvé.")
        if st.button("Entraîner rapidement le modèle sur donnees_test.csv"):
            if Path("donnees_test.csv").exists():
                with st.spinner("Entraînement du modèle en cours..."):
                    df_train = charger("donnees_test.csv")
                    X = df_train[COLONNES_TD].to_numpy()
                    y = df_train["glaucome"].to_numpy()
                    g = df_train["patient_id"].to_numpy()
                    params, _, poids = regler_hyperparametres(X, y, g, n_essais=5)
                    m = creer_xgb(params, poids).fit(X, y)
                    DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
                    m.save_model(CHEMIN_MODELE)
                    st.cache_resource.clear()
                st.success("Modèle entraîné et sauvegardé ! Rechargez ou passez à l'étape 2.")
            else:
                st.error("Le fichier `donnees_test.csv` est introuvable dans le dossier.")


# ---------------------------------------------------------------- 2. IMPORTER UN EXAMEN
elif menu == "2. Importer un examen":
    st.title("Importer un examen de champ visuel")

    onglet_csv, onglet_pdf = st.tabs(["📄 Fichier CSV (actif)", "📑 Rapport PDF Humphrey 24-2 (bientôt)"])

    with onglet_csv:
        st.write("Importez un fichier CSV contenant les colonnes `patient_id`, `oeil` (OD/OG) et `TD_01` à `TD_52`.")
        fichier_upload = st.file_uploader("Choisir un fichier CSV", type=["csv"])

        utiliser_demo = st.checkbox("Ou utiliser le fichier d'exemple `donnees_test.csv` présent dans le dossier")

        df_importe = None
        if fichier_upload is not None:
            df_importe = pd.read_csv(fichier_upload, sep=None, engine="python", dtype=str)
        elif utiliser_demo and Path("donnees_test.csv").exists():
            df_importe = pd.read_csv("donnees_test.csv", sep=None, engine="python", dtype=str)

        if df_importe is not None:
            manquantes = [c for c in ["patient_id", "oeil", *COLONNES_TD] if c not in df_importe.columns]
            if manquantes:
                st.error(f"Colonnes manquantes dans le fichier : {manquantes}")
            else:
                for c in COLONNES_TD:
                    df_importe[c] = nombre(df_importe[c])
                df_importe["oeil"] = df_importe["oeil"].astype(str).str.upper().str.strip()
                df_importe["patient_id"] = df_importe["patient_id"].astype(str).str.strip()
                df_importe = df_importe.dropna(subset=COLONNES_TD).copy()

                st.session_state["df_examen"] = df_importe
                st.success(
                    f"✅ {len(df_importe)} examen(s) chargé(s) pour {df_importe['patient_id'].nunique()} patient(s)."
                )
                st.dataframe(df_importe[["patient_id", "oeil"] + COLONNES_TD[:6]].head(10), use_container_width=True)
                st.info("Rendez-vous maintenant dans l'onglet **3. Analyser**.")

    with onglet_pdf:
        st.info(
            "L'extraction automatique des rapports PDF Humphrey 24-2 sera intégrée dès réception d'un exemple de rapport anonymisé."
        )
        st.file_uploader("Déposer un rapport PDF 24-2 (aperçu)", type=["pdf"], disabled=True)


# ---------------------------------------------------------------- 3. ANALYSER
elif menu == "3. Analyser":
    st.title("Lancer l'analyse")

    if modele is None:
        st.error("Aucun modèle disponible. Retournez à l'Accueil pour entraîner le modèle.")
    elif "df_examen" not in st.session_state:
        st.warning("Aucun examen importé. Allez d'abord dans **2. Importer un examen**.")
    else:
        df_ex = st.session_state["df_examen"].copy()
        patients = sorted(df_ex["patient_id"].unique())
        patient_choisi = st.selectbox("Sélectionner un patient à analyser :", patients)

        sous_df = df_ex[df_ex["patient_id"] == patient_choisi].copy()
        st.write(f"Examens disponibles pour **{patient_choisi}** : {', '.join(sous_df['oeil'].tolist())}")

        if st.button("🔍 Lancer l'analyse maintenant", type="primary"):
            resultats_patient = []
            lignes_histo = []

            for _, ligne in sous_df.iterrows():
                oeil = ligne["oeil"]
                valeurs_brutes = ligne[COLONNES_TD].to_numpy(dtype=float)
                # Réorientation OG -> OD conforme au protocole
                valeurs_od = og_vers_od(valeurs_brutes) if oeil == "OG" else valeurs_brutes

                X_mat = valeurs_od.reshape(1, -1)
                proba = float(modele.predict_proba(X_mat)[0, 1])
                contribs = modele.get_booster().predict(xgb.DMatrix(X_mat), pred_contribs=True)[0, :-1]

                titre_risque, niveau, explication = interpreter_risque(proba, seuils)
                md_estime = float(np.mean(valeurs_od))

                resultats_patient.append(
                    {
                        "patient_id": patient_choisi,
                        "oeil": oeil,
                        "proba": proba,
                        "titre_risque": titre_risque,
                        "niveau": niveau,
                        "explication": explication,
                        "md_estime": md_estime,
                        "valeurs_od": valeurs_od,
                        "contribs": contribs,
                    }
                )
                lignes_histo.append(
                    {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "patient_id": patient_choisi,
                        "oeil": oeil,
                        "niveau_risque": titre_risque,
                        "indice_ia_pct": round(proba * 100, 1),
                        "deviation_moyenne_dB": round(md_estime, 2),
                    }
                )

            st.session_state["derniers_resultats"] = resultats_patient
            enregistrer_historique(lignes_histo)
            st.success("Analyse terminée ! Consultez l'onglet **4. Résultats compréhensibles**.")


# ---------------------------------------------------------------- 4. RÉSULTATS COMPRÉHENSIBLES
elif menu == "4. Résultats compréhensibles":
    st.title("Résultats de l'analyse")

    if "derniers_resultats" not in st.session_state:
        st.info("Aucune analyse récente. Importez un examen et cliquez sur **Analyser**.")
    else:
        res_liste = st.session_state["derniers_resultats"]
        st.subheader(f"Patient : {res_liste[0]['patient_id']}")

        for r in res_liste:
            st.divider()
            nom_oeil = "Œil Droit (OD)" if r["oeil"] == "OD" else "Œil Gauche (OG)"
            st.markdown(f"### {nom_oeil}")

            # 1. Synthèse claire sans jargon
            if r["niveau"] == "error":
                st.error(f"**{r['titre_risque']}**\n\n{r['explication']}")
            elif r["niveau"] == "warning":
                st.warning(f"**{r['titre_risque']}**\n\n{r['explication']}")
            else:
                st.success(f"**{r['titre_risque']}**\n\n{r['explication']}")

            # 2. Détails complémentaires pour le médecin
            with st.expander("🩺 Détails cliniques et carte du champ visuel (pour le médecin)", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Déviation moyenne calculée (52 pts)", f"{r['md_estime']:.2f} dB")
                c2.metric("Score de compatibilité glaucome", f"{r['proba'] * 100:.1f} %")

                fig = dessiner_carte_cv(
                    r["valeurs_od"],
                    contribs=np.maximum(r["contribs"], 0),
                    titre=f"{nom_oeil} — Réorienté selon le référentiel anatomique OD",
                )
                st.pyplot(fig)
                plt.close(fig)


# ---------------------------------------------------------------- 5. HISTORIQUE & RAPPORTS
elif menu == "5. Historique & Rapports":
    st.title("Historique des examens & Rapports")

    if CHEMIN_HISTORIQUE.exists():
        df_h = pd.read_csv(CHEMIN_HISTORIQUE)
        st.dataframe(df_h.iloc[::-1], use_container_width=True)

        csv_export = df_h.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger l'historique complet (CSV)",
            data=csv_export,
            file_name="historique_glaucome_24_2.csv",
            mime="text/csv",
        )
    else:
        st.info("Aucun examen n'a encore été enregistré dans l'historique.")

    if (DOSSIER_RESULTATS / "rapport.txt").exists():
        with st.expander("📊 Rapport scientifique du modèle entraîné"):
            st.text((DOSSIER_RESULTATS / "rapport.txt").read_text(encoding="utf-8"))