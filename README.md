# Rap humain vs. Rap IA — Analyse stylistique et lexicale comparative

Projet du cours **Analyse Statistique des Données Textuelles** — M1 TAL  
Anna Sugak, Anna Ushmarina, Daria Tupikina, Elane Grandmougin, Serafima Klimova

---

## Description

Ce projet compare un corpus de paroles de rap américain collectées via Genius avec un corpus de paroles générées par trois modèles de langage — **Gemma**, **Mixtral** et **Qwen** — autour de trois thèmes : **Money**, **Way of Life** et **Love**.

L'objectif est d'analyser dans quelle mesure les structures stylistiques et les biais sociaux (notamment la misogynie) présents dans les œuvres humaines sont reproduits par l'IA.

---

## Structure du dépôt

```
ASDT_projet/
├── code/
│   ├── genius_album.py       # Collecte les paroles d'un album via l'API Genius + iTunes
│   ├── genius_artist.py      # Collecte les paroles d'un artiste via l'API Genius
│   ├── search_song.py        # Collecte les paroles depuis une liste "Artiste - Titre"
│   ├── check_duplicates.py   # Détecte les doublons entre deux listes de chansons
│   ├── word_count.py         # Compte le nombre de mots dans un fichier
│   ├── analyze_rimes.py      # Calcule la densité de rimes et les schémas (ABCD, AABB…)
│   ├── burrows_delta.py      # Calcule la distance stylistique (Delta de Burrows) et génère une heatmap
│   ├── analyse_ttr.py        # Calcule le Type-Token Ratio par thème et par source
│   ├── analyse_sentiment.py  # Analyse de sentiment VADER par thème et par source
│   └── analyse_lexicale.py   # Fréquences des termes genrés/misogynes, cooccurrences, analyse spaCy
├── corpus/
│   ├── humain/               # Paroles Genius organisées par thème (life/, love/, money/)
│   └── IA/                   # Paroles générées (gemma/, mixtral/, qwen/)
└── outputs/                  # Graphiques et rapports générés automatiquement
```

---

## Installation

```bash
pip install lyricsgenius pronouncing nltk pandas matplotlib seaborn \
            networkx wordcloud spacy scikit-learn vaderSentiment
python -m spacy download en_core_web_sm
```

Les scripts de collecte nécessitent un token d'accès Genius :

```bash
export GENIUS_ACCESS_TOKEN=votre_token_ici
```

---

## Utilisation

### Collecte du corpus humain

```bash
# Paroles d'un artiste (15 chansons les plus populaires)
python code/genius_artist.py "Kendrick Lamar"

# Paroles d'un album complet
python code/genius_album.py "Drake" "Take Care"

# Paroles depuis une liste "Artiste - Titre" (une par ligne)
python code/search_song.py songs.txt corpus_output.txt

# Vérification des doublons entre deux listes
python code/check_duplicates.py liste1.txt liste2.txt
```

> Un nettoyage manuel est nécessaire après la collecte : suppression des métadonnées Genius, des annotations `[Verse 1]`, `[Chorus]`, etc.

### Analyses

```bash
python code/word_count.py corpus/humain/life/life_corpus.txt
python code/analyze_rimes.py        # → affichage console
python code/burrows_delta.py        # → outputs/delta.png
python code/analyse_ttr.py          # → ttr_results.png
python code/analyse_sentiment.py    # → sentiment_results.png
python code/analyse_lexicale.py     # → outputs/ (graphiques + rapports texte)
```

---

## Résultats

### Schémas de rimes

Les textes humains se caractérisent par une grande liberté structurelle : le schéma ABCD (aucune rime) est dominant (6 162 occurrences). Les modèles IA produisent des schémas beaucoup plus réguliers, Qwen privilégiant massivement AAAA (toutes les lignes riment).

| Source  | Densité de rimes | Schéma dominant |
|---------|-----------------|-----------------|
| Humain  | 28 %            | ABCD            |
| Mistral | 35 %            | ABCD            |
| Gemma   | 45 %            | ABCC / ABBC     |
| Qwen    | 68 %            | AAAA            |

### Type-Token Ratio (richesse lexicale)

Les modèles IA affichent systématiquement un TTR supérieur au corpus humain. Ce résultat contre-intuitif s'explique par la nature du rap : répétitions et refrains sont des procédés rhétoriques intentionnels. Un TTR élevé signale un éloignement du genre, pas une meilleure qualité.

### Sentiment (VADER)

Sur le thème Money, les modèles IA affichent des scores clairement positifs (Mixtral : +0,22 ; Gemma et Qwen : +0,10), tandis que le corpus humain reste neutre. En neutralisant le vocabulaire offensant, l'IA produit mécaniquement des textes au ton plus positif, créant une image édulcorée du genre.

### Distance stylistique (Delta de Burrows)

La heatmap révèle une séparation nette entre les deux sous-corpus. La distance minimale humain–IA (1,06) est supérieure à la distance maximale intra-humain (0,64) : l'algorithme distingue les deux sources sans ambiguïté. Qwen est le modèle stylistiquement le plus proche des textes humains malgré ses filtres de sécurité plus stricts.

| Comparaison       | Delta       |
|-------------------|-------------|
| Intra-humain      | 0,43 – 0,64 |
| Humain – Qwen     | ~1,10       |
| Humain – Gemma    | ~1,49       |
| Humain – Mistral  | ~1,59       |

### Termes genrés et misogynie

L'IA n'évite pas seulement le registre offensant : elle réduit globalement la présence des femmes dans les textes. Quand elles apparaissent, c'est presque exclusivement dans un registre valorisant, ce qui inverse le rapport observé dans le corpus humain.

| Catégorie de termes | Humain    | IA       |
|---------------------|-----------|----------|
| Misogynes           | 2 638 /M  | 34 /M    |
| Neutres genrés      | 10 434 /M | 1 790 /M |
| Positifs            | 199 /M    | 165 /M   |

---

## Conclusion

L'IA reproduit les structures thématiques du rap (argent, amour, vie difficile) mais pas sa réalité culturelle et linguistique. Le corpus généré est « nettoyé » : absence de registre offensant, ton plus positif, figures féminines invisibilisées. L'IA imite la forme du rap, mais pas son fond.