import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# 1. Chargement des données

def charger_corpus(chemin_base):
    """
    Parcourt tous les sous-dossiers du chemin donné et retourne
    un dictionnaire : {"humain/life": "texte complet...", ...}

    Chaque dossier terminal devient un "auteur" dans l'analyse.
    Tous les fichiers .txt du dossier sont concaténés en un seul texte.
    """
    corpus = {}

    for racine, dossiers, fichiers in os.walk(chemin_base):
        # On ne garde que les dossiers contenant des .txt
        fichiers_txt = [f for f in fichiers if f.endswith(".txt")]
        if not fichiers_txt:
            continue

        # Construction de l'étiquette à partir du chemin relatif
        # ex. : "humain/life" ou "IA/qwen"
        chemin_relatif = os.path.relpath(racine, chemin_base)
        etiquette = chemin_relatif.replace(os.sep, "/")

        # On ignore le dossier all songs (listes de chansons, pas des paroles)
        if "all songs" in etiquette:
            continue

        # Lecture et concaténation de tous les fichiers du dossier
        textes = []
        for nom_fichier in fichiers_txt:
            chemin_fichier = os.path.join(racine, nom_fichier)
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                textes.append(f.read())

        corpus[etiquette] = " ".join(textes)

    return corpus

# Le script se trouve dans /code, le corpus est un niveau au-dessus
corpus = charger_corpus("../corpus")

print("Groupes chargés :")
for etiquette, texte in corpus.items():
    print(f"  {etiquette} : {len(texte.split())} mots")

# 2. Séparation des groupes

# Groupes humains (3 thèmes : love, life, money)
etiquettes_humain = [k for k in corpus if k.startswith("humain/")]

# Groupes IA (3 modèles : gemma, mixtral, qwen)
etiquettes_ia = [k for k in corpus if k.startswith("IA/")]

# 3. Calcul de la Delta de Burrows

def calculer_delta(dict_docs, n_features=200):
    """
    Calcule la distance de Burrows (Burrows' Delta) entre tous les groupes.

    Principe : deux textes stylistiquement proches auront des fréquences
    de mots similaires. Delta mesure à quel point elles diffèrent.

    Étapes :
    1. Vectorisation — compter les N mots les plus fréquents du corpus
    2. Fréquences relatives — diviser par la longueur pour neutraliser
       l'effet de taille (un texte 3x plus long n'est pas 3x différent)
    3. Z-score — normaliser chaque mot pour que les mots rares et fréquents
       contribuent également à la distance finale
    4. Distance de Manhattan — moyenne des écarts absolus entre les vecteurs
    """
    etiquettes = list(dict_docs.keys())
    textes = list(dict_docs.values())

    # Étape 1 : fréquences brutes des N mots les plus courants
    vectoriseur = CountVectorizer(max_features=n_features)
    matrice = vectoriseur.fit_transform(textes).toarray().astype(float)
    df = pd.DataFrame(matrice, index=etiquettes,
                      columns=vectoriseur.get_feature_names_out())

    # Étape 2 : fréquences relatives (chaque ligne divisée par sa somme)
    df = df.div(df.sum(axis=1), axis=0)

    # Étape 3 : z-score colonne par colonne
    # +1e-9 pour éviter la division par zéro si un mot est identique partout
    df_z = (df - df.mean()) / (df.std() + 1e-9)

    # Étape 4 : matrice de distances de Manhattan entre toutes les paires
    n = len(etiquettes)
    matrice_distances = pd.DataFrame(index=etiquettes, columns=etiquettes,
                                     dtype=float)

    for i in range(n):
        for j in range(n):
            delta = np.mean(np.abs(df_z.iloc[i] - df_z.iloc[j]))
            matrice_distances.iloc[i, j] = delta

    return matrice_distances

# 4. Analyse 1 — Carte thermique de tous les groupes

# 4. Analyse 1 — Carte thermique de tous les groupes

# Création du dossier outputs s'il n'existe pas encore
dossier_outputs = "../outputs"
os.makedirs(dossier_outputs, exist_ok=True)  # exist_ok=True évite une erreur si le dossier existe déjà

distances_global = calculer_delta(corpus, n_features=200)

plt.figure(figsize=(14, 10))
sns.heatmap(
    distances_global.astype(float),
    annot=True, fmt=".2f",
    cmap="YlOrRd_r",
    linewidths=0.5
)
plt.title("Burrows' Delta — tous les groupes\n(plus foncé = stylistiquement plus proche)")
plt.tight_layout()
plt.savefig(os.path.join(dossier_outputs, "delta.png"), dpi=150) 
plt.show()

# 5. Analyse 2 — Attribution : vers quel groupe humainchaque modèle IA est-il attiré ?

# On regroupe humains + IA pour calculer leurs distances mutuelles
tous_docs = {k: corpus[k] for k in etiquettes_humain + etiquettes_ia}
distances_attr = calculer_delta(tous_docs, n_features=200)

print("\n=== Attribution : à quel thème humain ressemble chaque groupe IA ? ===\n")

resultats_attribution = []
for etiquette_ia in etiquettes_ia:
    # Distance de ce groupe IA vers chaque groupe humain
    distances_vers_humains = {
        h: float(distances_attr.loc[etiquette_ia, h])
        for h in etiquettes_humain
    }
    # Le groupe humain le plus proche = distance minimale
    plus_proche = min(distances_vers_humains, key=distances_vers_humains.get)

    resultats_attribution.append({
        "Groupe IA":     etiquette_ia,
        "Attribué à":    plus_proche,
        "Delta":         round(distances_vers_humains[plus_proche], 4),
    })

df_resultats = pd.DataFrame(resultats_attribution)
print(df_resultats.to_string(index=False))

# 6. Analyse 3 — Quel modèle est le plus proche des humains ?

print("\n=== Distance moyenne de chaque modèle vers les humains ===\n")
print("(plus faible = stylistiquement plus proche des textes humains)\n")

for modele in ["qwen", "gemma", "mixtral"]:
    # On récupère toutes les étiquettes IA contenant le nom du modèle
    etiquettes_modele = [k for k in etiquettes_ia if modele in k]

    distances = []
    for ia in etiquettes_modele:
        for h in etiquettes_humain:
            distances.append(float(distances_attr.loc[ia, h]))

    print(f"  {modele} : Delta moyenne = {np.mean(distances):.4f}")