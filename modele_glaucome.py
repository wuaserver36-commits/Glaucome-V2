"""
Détection précoce du glaucome par IA à partir du champ visuel Humphrey 24-2.
Programme conforme au protocole : modèle principal XGBoost sur les 52 valeurs
de déviation totale, comparé à une régression logistique (MD + PSD), à un
random forest, au GHT et aux lecteurs humains.

Utilisation :
    python modele_glaucome.py mes_donnees.csv --essais 100 --sortie resultats

Colonnes obligatoires du fichier CSV (une ligne par œil) :
    patient_id, oeil (OD ou OG), glaucome (0 = normal, 1 = glaucome débutant),
    TD_01 ... TD_52 (déviation totale en dB, voir grille_24_2.py pour l'ordre)
Colonnes facultatives :
    MD, PSD                     -> régression logistique de référence
    GHT                         -> "normal", "limite" ou "hors limites"
    lecteur_1, lecteur_2, ...   -> avis des lecteurs humains (0 ou 1)
    age, sexe, atcd_familial, pachymetrie -> modèle secondaire exploratoire
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grille_24_2 import COLONNES_TD, POINTS_OD, TACHE_AVEUGLE_OD, og_vers_od

GRAINE = 2026          # graine aléatoire fixe : résultats reproductibles
PART_TEST = 0.20       # 20 % des patients mis de côté pour l'examen final
CLINIQUE = ["age", "sexe_f", "atcd_familial", "pachymetrie"]


# ---------------------------------------------------------------- 1. Données
def nombre(serie):
    """Convertit une colonne en nombres, en acceptant la virgule décimale."""
    return pd.to_numeric(serie.astype(str).str.replace(",", ".").str.strip(), errors="coerce")


def code_ght(valeur):
    """0 = normal, 1 = limite, 2 = hors limites normales, NaN si absent."""
    if pd.isna(valeur):
        return np.nan
    v = str(valeur).lower()
    if "hors" in v or "outside" in v:
        return 2
    if "dans" in v or "within" in v or "normal" in v:
        return 0
    if "limite" in v or "border" in v:
        return 1
    return np.nan


def charger(chemin):
    df = pd.read_csv(chemin, sep=None, engine="python", dtype=str)
    manquantes = [c for c in ["patient_id", "oeil", "glaucome", *COLONNES_TD] if c not in df]
    if manquantes:
        raise SystemExit(f"Colonnes manquantes dans le fichier : {manquantes}")

    for c in ["glaucome", *COLONNES_TD, "MD", "PSD", "age", "atcd_familial", "pachymetrie"]:
        if c in df:
            df[c] = nombre(df[c])
    for c in [c for c in df if c.startswith("lecteur_")]:
        df[c] = nombre(df[c])
    df["oeil"] = df["oeil"].str.upper().str.strip()

    avant = len(df)
    df["patient_id"] = df["patient_id"].fillna("").astype(str).str.strip()
    df = df[
        df["patient_id"].ne("")
        & df["oeil"].isin(["OD", "OG"])
        & df["glaucome"].isin([0, 1])
    ].copy()
    df = df.dropna(subset=COLONNES_TD).copy()
    if len(df) < avant:
        print(f"  {avant - len(df)} ligne(s) écartée(s) : identifiant, œil, diagnostic ou valeurs TD manquants.")

    doublons = df.duplicated(subset=["patient_id", "oeil"], keep=False)
    if doublons.any():
        exemples = df.loc[doublons, ["patient_id", "oeil"]].head(10).to_dict("records")
        raise SystemExit(
            "Doublons détectés : un seul examen par patient et par œil est attendu. "
            f"Exemples : {exemples}"
        )

    og = df["oeil"] == "OG"
    if og.any():
        df.loc[og, COLONNES_TD] = og_vers_od(df.loc[og, COLONNES_TD].to_numpy())

    df["glaucome"] = df["glaucome"].astype(int)
    if "sexe" in df:
        df["sexe_f"] = (df["sexe"].astype(str).str.upper().str[0] == "F").astype(int)
    if "GHT" in df:
        df["GHT_code"] = df["GHT"].map(code_ght)
    return df


# ---------------------------------------------------------------- 2. Modèles
def creer_xgb(params, poids_positifs):
    return xgb.XGBClassifier(**params, scale_pos_weight=poids_positifs,
                             eval_metric="logloss", tree_method="hist",
                             random_state=GRAINE, n_jobs=-1)


def predictions_oof(modele, X, y, groupes, n_splits=5):
    """Prédictions hors pli : les deux yeux d'un patient restent toujours ensemble."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=GRAINE)
    predictions = np.full(len(y), np.nan, dtype=float)
    for indices_app, indices_val in cv.split(X, y, groups=groupes):
        modele_pli = modele()
        modele_pli.fit(X[indices_app], y[indices_app])
        predictions[indices_val] = modele_pli.predict_proba(X[indices_val])[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Certaines observations n'ont pas reçu de prédiction hors pli.")
    return predictions


def regler_hyperparametres(X, y, groupes, n_essais):
    poids = (y == 0).sum() / max((y == 1).sum(), 1)

    def objectif(essai):
        params = {
            "max_depth": essai.suggest_int("max_depth", 2, 6),
            "learning_rate": essai.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": essai.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample": essai.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": essai.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": essai.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": essai.suggest_float("reg_lambda", 0.0, 10.0),
        }
        p = predictions_oof(lambda: creer_xgb(params, poids), X, y, groupes, n_splits=5)
        return roc_auc_score(y, p)

    def suivi(etude, essai):
        if (essai.number + 1) % 10 == 0:
            print(f"  essai {essai.number + 1}/{n_essais} - meilleure AUC : {etude.best_value:.3f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    etude = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=GRAINE)
    )
    etude.optimize(objectif, n_trials=n_essais, callbacks=[suivi])
    return etude.best_params, etude.best_value, poids


def seuils_decision(y, p):
    """Seuils fixés sur la validation croisée, puis appliqués tels quels au test."""
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.minimum(thr, 1.0)
    spec = 1 - fpr
    seuils = {}
    for cible in (0.95, 0.90):
        i = np.argmax(np.where(spec >= cible, tpr, -1))
        seuils[f"specificite_{int(cible * 100)}"] = float(thr[i])
    seuils["youden"] = float(thr[np.argmax(tpr - fpr)])
    return seuils


# ---------------------------------------------------------------- 3. Statistiques
def ic_wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    marge = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centre - marge, centre + marge)


def bootstrap_groupes(y, groupes, n=2000, graine=GRAINE):
    """Tire des patients avec remise et conserve ensemble leurs deux yeux."""
    rng = np.random.default_rng(graine)
    groupes = np.asarray(groupes)
    uniques = np.unique(groupes)
    indices_par_groupe = {g: np.flatnonzero(groupes == g) for g in uniques}
    for _ in range(n):
        tires = rng.choice(uniques, size=len(uniques), replace=True)
        yield np.concatenate([indices_par_groupe[g] for g in tires])


def sens_spec(y, positif, groupes=None, n_boot=1000):
    y, positif = np.asarray(y), np.asarray(positif)
    vp = int(((positif == 1) & (y == 1)).sum())
    n_pos = int((y == 1).sum())
    vn = int(((positif == 0) & (y == 0)).sum())
    n_neg = int((y == 0).sum())
    resultat = {
        "sensibilite": vp / n_pos if n_pos else float("nan"),
        "specificite": vn / n_neg if n_neg else float("nan"),
    }
    if groupes is not None:
        sens, spec = [], []
        for idx in bootstrap_groupes(y, groupes, n=n_boot):
            yy, pp = y[idx], positif[idx]
            if (yy == 1).any():
                sens.append(((pp == 1) & (yy == 1)).sum() / (yy == 1).sum())
            if (yy == 0).any():
                spec.append(((pp == 0) & (yy == 0)).sum() / (yy == 0).sum())
        resultat["sensibilite_IC95_cluster"] = (
            tuple(np.percentile(sens, [2.5, 97.5])) if sens else (np.nan, np.nan)
        )
        resultat["specificite_IC95_cluster"] = (
            tuple(np.percentile(spec, [2.5, 97.5])) if spec else (np.nan, np.nan)
        )
    return resultat


def auc_bootstrap(y, p, groupes, n=2000):
    y, p = np.asarray(y), np.asarray(p)
    valeurs = []
    for idx in bootstrap_groupes(y, groupes, n=n):
        if len(np.unique(y[idx])) == 2:
            valeurs.append(roc_auc_score(y[idx], p[idx]))
    if not valeurs:
        return roc_auc_score(y, p), (float("nan"), float("nan"))
    return roc_auc_score(y, p), tuple(np.percentile(valeurs, [2.5, 97.5]))


def kappa_fleiss(avis):
    avis = np.asarray(avis); n, r = avis.shape
    positifs = avis.sum(axis=1)
    comptes = np.stack([r - positifs, positifs], axis=1)
    p_i = ((comptes**2).sum(axis=1) - r) / (r * (r - 1))
    p_j = comptes.sum(axis=0) / (n * r)
    return float((p_i.mean() - (p_j**2).sum()) / (1 - (p_j**2).sum()))


# ---------------------------------------------------------------- 4. Figures
def figure_roc(y, courbes, points, chemin):
    fig, ax = plt.subplots(figsize=(6, 6))
    for nom, p in courbes.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{nom} (AUC {roc_auc_score(y, p):.2f})")
    for nom, (fpr, tpr) in points.items():
        ax.plot(fpr, tpr, "o", label=nom)
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("1 - spécificité"); ax.set_ylabel("Sensibilité")
    ax.set_title("Courbes ROC sur le jeu de test"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(chemin, dpi=150); plt.close(fig)


def figure_calibration(y, p, chemin):
    obs, pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pred, obs, "o-", label="XGBoost (validation croisée)")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="Calibration parfaite")
    ax.set_xlabel("Probabilité prédite"); ax.set_ylabel("Proportion observée de glaucomes")
    ax.set_title("Calibration du modèle"); ax.legend()
    fig.tight_layout(); fig.savefig(chemin, dpi=150); plt.close(fig)


def figure_importance(importance, chemin):
    """Carte des points les plus discriminants, en orientation œil droit."""
    fig, ax = plt.subplots(figsize=(7, 6))
    pts = np.array(POINTS_OD)
    nuage = ax.scatter(pts[:, 0], pts[:, 1], c=importance, cmap="Reds", s=900, marker="s",
                       edgecolors="grey")
    for (x, y), v in zip(POINTS_OD, importance):
        ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7)
    for x, y in TACHE_AVEUGLE_OD:
        ax.scatter(x, y, s=900, marker="s", color="black")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-32, 32); ax.set_ylim(-26, 26); ax.set_aspect("equal")
    ax.set_xlabel("Nasal  <-  degrés  ->  Temporal")
    ax.set_title("Importance des points (|SHAP| moyen) - orientation œil droit")
    fig.colorbar(nuage, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(chemin, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 5. Programme
def main():
    ap = argparse.ArgumentParser(description="Détection du glaucome débutant par XGBoost (CV 24-2)")
    ap.add_argument("fichier", help="fichier CSV des données")
    ap.add_argument("--essais", type=int, default=100, help="essais de réglage (défaut : 100)")
    ap.add_argument("--sortie", default="resultats", help="dossier des résultats")
    a = ap.parse_args()
    if a.essais < 1:
        ap.error("--essais doit être supérieur ou égal à 1")
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    print("1. Lecture des données")
    df = charger(a.fichier)
    n_patients = df["patient_id"].nunique()
    n_glauc_yeux = int(df["glaucome"].sum())
    print(
        f"  {n_patients} patients, {len(df)} yeux/examens retenus : "
        f"{(df['oeil'] == 'OD').sum()} OD, {(df['oeil'] == 'OG').sum()} OG ; "
        f"{len(df) - n_glauc_yeux} yeux sans glaucome, {n_glauc_yeux} yeux glaucomateux"
    )
    if df["glaucome"].nunique() != 2:
        raise SystemExit("Il faut des examens des deux classes : glaucome=0 et glaucome=1.")
    effectifs_groupes = df.groupby("patient_id")["glaucome"].nunique()
    print(
        f"  {int((effectifs_groupes > 1).sum())} patient(s) ont des diagnostics différents "
        "entre les deux yeux ; les étiquettes restent définies par œil."
    )

    print("2. Séparation apprentissage/test par patient (sans chevauchement)")
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=GRAINE)
    idx_app, idx_test = next(
        split.split(df[COLONNES_TD], df["glaucome"], groups=df["patient_id"])
    )
    app, test = df.iloc[idx_app].copy(), df.iloc[idx_test].copy()
    patients_app = set(app["patient_id"])
    patients_test = set(test["patient_id"])
    if patients_app & patients_test:
        raise RuntimeError("Fuite de patients détectée entre apprentissage et test.")
    X_app, y_app = app[COLONNES_TD].to_numpy(), app["glaucome"].to_numpy()
    X_test, y_test = test[COLONNES_TD].to_numpy(), test["glaucome"].to_numpy()
    g_app, g_test = app["patient_id"].to_numpy(), test["patient_id"].to_numpy()
    if len(np.unique(y_app)) != 2 or len(np.unique(y_test)) != 2:
        raise SystemExit(
            "La séparation a produit un ensemble sans les deux classes. "
            "Vérifiez le nombre de patients et la répartition des diagnostics."
        )
    print(
        f"  apprentissage : {len(patients_app)} patients / {len(app)} yeux ; "
        f"test : {len(patients_test)} patients / {len(test)} yeux"
    )

    print(f"3. Réglage des hyperparamètres ({a.essais} essais, CV groupée en 5 plis)")
    params, auc_cv, poids = regler_hyperparametres(X_app, y_app, g_app, a.essais)

    print("4. Seuils de décision et calibration (prédictions hors pli groupées)")
    p_oof = predictions_oof(lambda: creer_xgb(params, poids), X_app, y_app, g_app, n_splits=5)
    seuils = seuils_decision(y_app, p_oof)
    figure_calibration(y_app, p_oof, sortie / "calibration.png")

    print("5. Entraînement final et examen sur le jeu de test")
    modele = creer_xgb(params, poids).fit(X_app, y_app)
    modele.save_model(sortie / "modele_xgboost.json")
    p_test = modele.predict_proba(X_test)[:, 1]
    auc, ic = auc_bootstrap(y_test, p_test, g_test)
    res = {
        "avertissement": "Données simulées : aucune valeur clinique.",
        "unite_analyse": "œil, avec séparation et bootstrap regroupés par patient",
        "effectifs": {
            "patients": int(n_patients), "yeux": int(len(df)),
            "yeux_glaucome": n_glauc_yeux,
            "patients_apprentissage": len(patients_app), "yeux_apprentissage": len(app),
            "patients_test": len(patients_test), "yeux_test": len(test),
            "yeux_OD": int((df["oeil"] == "OD").sum()),
            "yeux_OG": int((df["oeil"] == "OG").sum()),
        },
        "hyperparametres": params,
        "AUC_validation_croisee_groupes": auc_cv,
        "seuils": seuils,
        "brier_validation_croisee": brier_score_loss(y_app, p_oof),
        "XGBoost": {
            "AUC": auc, "AUC_IC95_bootstrap_patient": ic,
            **{
                nom: sens_spec(y_test, (p_test >= seuil).astype(int), g_test)
                for nom, seuil in seuils.items()
            },
        },
        "comparateurs": {},
        "note_statistique": (
            "IC par bootstrap de patients : les deux yeux d'un patient sont rééchantillonnés ensemble. "
            "Les p-values DeLong/McNemar classiques sont retirées car elles ne tiennent pas compte "
            "de la corrélation entre les deux yeux."
        ),
    }
    courbes = {"XGBoost (52 points)": p_test}
    points = {}

    # Comparateur 1 : régression logistique MD + PSD
    if {"MD", "PSD"} <= set(df) and app[["MD", "PSD"]].notna().any(axis=None):
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
        lr.fit(app[["MD", "PSD"]], y_app)
        p_lr = lr.predict_proba(test[["MD", "PSD"]])[:, 1]
        auc_lr, ic_lr = auc_bootstrap(y_test, p_lr, g_test)
        res["comparateurs"]["Logistique MD+PSD"] = {
            "AUC": auc_lr, "AUC_IC95_bootstrap_patient": ic_lr
        }
        courbes["Logistique MD + PSD"] = p_lr

    # Comparateur 2 : random forest sur les 52 points
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=GRAINE, n_jobs=-1
    ).fit(X_app, y_app)
    p_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf, ic_rf = auc_bootstrap(y_test, p_rf, g_test)
    res["comparateurs"]["Random forest"] = {
        "AUC": auc_rf, "AUC_IC95_bootstrap_patient": ic_rf
    }
    courbes["Random forest"] = p_rf

    # Comparateur 3 : GHT
    if "GHT_code" in test and test["GHT_code"].notna().all():
        for nom, seuil_ght in [("GHT (limite = négatif)", 2), ("GHT (limite = positif)", 1)]:
            pos = (test["GHT_code"] >= seuil_ght).astype(int).to_numpy()
            r = sens_spec(y_test, pos, g_test)
            res["comparateurs"][nom] = r
            points[nom] = (1 - r["specificite"], r["sensibilite"])

    # Comparateur 4 : lecteurs humains
    lecteurs = sorted(c for c in test if c.startswith("lecteur_"))
    if lecteurs and test[lecteurs].notna().all().all():
        for c in lecteurs:
            avis = test[c].astype(int).to_numpy()
            r = sens_spec(y_test, avis, g_test)
            res["comparateurs"][c] = r
            points[c.replace("_", " ").capitalize()] = (1 - r["specificite"], r["sensibilite"])
        if len(lecteurs) >= 2:
            res["kappa_fleiss_lecteurs_descriptif"] = kappa_fleiss(test[lecteurs].astype(int))

    # Modèle secondaire exploratoire : 52 points + données cliniques simples
    cliniques = [c for c in CLINIQUE if c in df]
    if cliniques:
        cols = COLONNES_TD + cliniques
        m2 = creer_xgb(params, poids).fit(app[cols].to_numpy(dtype=float), y_app)
        p_m2 = m2.predict_proba(test[cols].to_numpy(dtype=float))[:, 1]
        auc_m2, ic_m2 = auc_bootstrap(y_test, p_m2, g_test)
        res["modele_secondaire"] = {
            "variables_ajoutees": cliniques, "AUC": auc_m2,
            "AUC_IC95_bootstrap_patient": ic_m2,
        }
        courbes["XGBoost + clinique (exploratoire)"] = p_m2

    print("6. Importance des points (valeurs SHAP)")
    contributions = modele.get_booster().predict(
        xgb.DMatrix(X_test), pred_contribs=True
    )[:, :-1]
    importance = np.abs(contributions).mean(axis=0)
    figure_importance(importance, sortie / "importance_points.png")
    figure_roc(y_test, courbes, points, sortie / "courbes_roc.png")
    pd.DataFrame({
        "colonne": COLONNES_TD,
        "x_deg": [p[0] for p in POINTS_OD],
        "y_deg": [p[1] for p in POINTS_OD],
        "importance_SHAP": importance,
    }).sort_values("importance_SHAP", ascending=False).to_csv(
        sortie / "importance_points.csv", index=False
    )

    with open(sortie / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    ecrire_rapport(res, sortie / "rapport.txt")
    print(f"\nTerminé. Résultats dans le dossier « {sortie} ».\n")
    print((sortie / "rapport.txt").read_text(encoding="utf-8"))


def ecrire_rapport(res, chemin):
    pct = lambda v: f"{100 * v:.1f} %"
    intervalle = lambda t: f"[{pct(t[0])} ; {pct(t[1])}]"
    e = res["effectifs"]
    x = res["XGBoost"]
    lignes = [
        "AVERTISSEMENT : données simulées. Résultats purement techniques, sans valeur clinique.",
        "Analyse au niveau de l'œil ; séparation et intervalles de confiance regroupés par patient.",
        "",
        "RAPPORT - Détection du glaucome débutant par XGBoost (CV 24-2)",
        "=" * 68,
        f"Patients : {e['patients']} ; yeux : {e['yeux']} (OD : {e['yeux_OD']}, OG : {e['yeux_OG']})",
        f"Yeux avec glaucome : {e['yeux_glaucome']}",
        f"Apprentissage : {e['patients_apprentissage']} patients / {e['yeux_apprentissage']} yeux ; "
        f"test : {e['patients_test']} patients / {e['yeux_test']} yeux",
        f"AUC validation croisée groupée : {res['AUC_validation_croisee_groupes']:.3f}",
        f"Score de Brier validation croisée : {res['brier_validation_croisee']:.3f}",
        "",
        "MODÈLE PRINCIPAL (jeu de test)",
        f"  AUC : {x['AUC']:.3f} ; IC95 bootstrap patient {intervalle(x['AUC_IC95_bootstrap_patient'])}",
    ]
    for nom in ("specificite_95", "specificite_90", "youden"):
        r = x[nom]
        lignes.append(
            f"  Seuil {nom} ({res['seuils'][nom]:.2f}) : sensibilité {pct(r['sensibilite'])} "
            f"{intervalle(r['sensibilite_IC95_cluster'])}, spécificité {pct(r['specificite'])} "
            f"{intervalle(r['specificite_IC95_cluster'])}"
        )
    lignes += ["", "COMPARATEURS (jeu de test)"]
    for nom, r in res["comparateurs"].items():
        if "AUC" in r:
            lignes.append(
                f"  {nom} : AUC {r['AUC']:.3f} ; "
                f"IC95 bootstrap patient {intervalle(r['AUC_IC95_bootstrap_patient'])}"
            )
        else:
            lignes.append(
                f"  {nom} : sensibilité {pct(r['sensibilite'])}, spécificité {pct(r['specificite'])} "
                "(IC95 bootstrap patient disponibles dans resultats.json)"
            )
    if "kappa_fleiss_lecteurs_descriptif" in res:
        lignes.append(
            f"  Accord entre lecteurs (kappa de Fleiss, descriptif) : "
            f"{res['kappa_fleiss_lecteurs_descriptif']:.2f}"
        )
    if "modele_secondaire" in res:
        m = res["modele_secondaire"]
        lignes += [
            "",
            "MODÈLE SECONDAIRE EXPLORATOIRE (52 points + " + ", ".join(m["variables_ajoutees"]) + ")",
            f"  AUC {m['AUC']:.3f} ; IC95 bootstrap patient "
            f"{intervalle(m['AUC_IC95_bootstrap_patient'])}",
        ]
    lignes += [
        "",
        "Les p-values DeLong/McNemar classiques ne sont pas calculées : elles supposent",
        "des observations indépendantes et ne sont pas adaptées ici aux deux yeux d'un patient.",
        "Figures : courbes_roc.png, calibration.png, importance_points.png",
    ]
    Path(chemin).write_text("\n".join(lignes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
