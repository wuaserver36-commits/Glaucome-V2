# IA de détection du glaucome débutant sur champ visuel 24-2

Programme conforme au protocole : un modèle XGBoost entraîné sur les 52 valeurs de déviation totale, comparé à une régression logistique (MD + PSD), à un random forest, au GHT et aux lecteurs humains.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `modele_glaucome.py` | Programme principal : entraînement, évaluation, figures et rapport |
| `grille_24_2.py` | Description de la grille 24-2 et conversion des yeux gauches |
| `genere_donnees_test.py` | Crée des données fictives pour essayer le programme |
| `donnees_test.csv` | Exemple de fichier, au format à remplir avec vos patients |

Les trois fichiers `.py` doivent rester dans le même dossier.

## 1. Installer (une seule fois)

Installez Python 3.10 ou plus récent (python.org), puis, dans un terminal :

```
pip install xgboost optuna scikit-learn scipy pandas matplotlib
```

## 2. Essayer avec les données fictives

```
python modele_glaucome.py donnees_test.csv --essais 20
```

Les résultats de cet essai n'ont aucune valeur scientifique : ils servent seulement à vérifier que tout fonctionne.

## 3. Préparer vos données

Une ligne par œil, en partant du fichier `donnees_test.csv` comme modèle (Excel, enregistré en CSV ; la virgule décimale est acceptée).

**Colonnes obligatoires**

- `patient_id` : code anonyme (par exemple P0001), identique pour les deux yeux d'un patient
- `oeil` : OD ou OG
- `glaucome` : 0 = normal, 1 = glaucome débutant, selon le diagnostic de référence des experts
- `TD_01` à `TD_52` : valeurs de la carte de déviation totale, en dB

**Ordre de saisie des 52 valeurs.** Lisez la carte de déviation totale telle qu'elle est imprimée, ligne par ligne de haut en bas, de gauche à droite, en sautant les 2 cases de la tache aveugle. Les lignes contiennent 4, 6, 8, 8, 8, 8, 6 et 4 valeurs. Faites-le de la même façon pour un œil gauche : le programme le remet lui-même dans l'orientation d'un œil droit.

**Colonnes facultatives**

- `MD`, `PSD` : pour la régression logistique de référence
- `GHT` : « normal », « limite » ou « hors limites »
- `lecteur_1`, `lecteur_2`, `lecteur_3` : avis des lecteurs humains (0 ou 1)
- `age`, `sexe` (H/F), `atcd_familial` (0/1), `pachymetrie` (µm) : modèle secondaire exploratoire

Les deux yeux sont conservés et analysés séparément. La séparation apprentissage/test, la validation croisée et les intervalles de confiance par bootstrap regroupent les observations par patient : les deux yeux d'une même personne ne se retrouvent jamais dans des ensembles différents.

## 4. Lancer l'analyse réelle

```
python modele_glaucome.py mes_donnees.csv --essais 100 --sortie resultats
```

Comptez de quelques minutes à une demi-heure selon l'ordinateur.

## 5. Ce que vous obtenez (dossier `resultats`)

| Fichier | Contenu |
|---|---|
| `rapport.txt` | Résumé en français : AUC, sensibilité, spécificité, comparaisons et valeurs p |
| `resultats.json` | Tous les chiffres, pour les tableaux de l'article |
| `courbes_roc.png` | Courbes ROC du modèle et des comparateurs |
| `calibration.png` | Concordance entre probabilités prédites et observées |
| `importance_points.png` | Carte des points du champ visuel les plus utiles au modèle |
| `importance_points.csv` | Les mêmes valeurs, en tableau |
| `modele_xgboost.json` | Le modèle entraîné, réutilisable pour de nouveaux patients |

## Points importants

- La graine aléatoire est fixée (2026) : relancer le programme sur les mêmes données donne les mêmes résultats.
- Le jeu de test (environ 20 % des patients) est constitué par validation croisée stratifiée et groupée ; les deux yeux d'un patient restent dans le même ensemble. N'ajustez pas le programme après avoir vu les résultats du test.
- Les seuils de décision (95 % et 90 % de spécificité, indice de Youden) sont fixés en validation croisée groupée, puis appliqués sans modification au jeu de test.\n- Les intervalles de confiance sont estimés par bootstrap de patients, en rééchantillonnant ensemble les deux yeux.\n- Les p-values classiques de DeLong et McNemar sont volontairement absentes : elles ne tiennent pas compte de la corrélation entre les deux yeux.
- Ne transmettez jamais de fichier contenant des noms ou des numéros de dossier : uniquement des codes anonymes.
