import os
import re
from collections import Counter

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# CONFIGURATION
HUMAN_CORPUS_DIR = "corpus/humain"
AI_CORPUS_DIR    = "corpus/IA"

HUMAN_THEMES = ["life", "love", "money"]
AI_MODELS    = ["gemma", "mixtral", "qwen"]
THEMES       = ["life", "love", "money"]

THEME_LABELS = {"life": "Way of Life", "love": "Love", "money": "Money"}

# LEXIQUE CATÉGORISÉ

LEXICON = {
    "misogyne": [
        "bitch", "hoe", "thot", "slut", "trick", "whore", "skank",
        "smash", "groupie", "ratchet", "bust down", "jump off", "side chick"
    ],
    "neutre_genré": [
        "girl", "woman", "female", "shorty", "lady", "chick",
        "honey", "baby", "shawty", "her", "she"
    ],
    "positif_genré": [
        "queen", "goddess", "wife", "wifey", "loyal", "real one"
    ]
}

ALL_TERMS = {term: cat for cat, terms in LEXICON.items() for term in terms}

#CHARGEMENT DU CORPUS

def load_txt_from_dir(directory):
    texts = []
    if not os.path.exists(directory):
        return ""
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".txt"):
            with open(os.path.join(directory, fname), encoding="utf-8") as f:
                texts.append(f.read())
    return "\n".join(texts)


def load_human_by_theme(base_dir, themes):
    """Retourne (texte_global, {theme: texte})."""
    by_theme = {}
    for theme in themes:
        text = load_txt_from_dir(os.path.join(base_dir, theme))
        if text:
            by_theme[theme] = text
            print(f"  [humain/{theme}] {len(text.split()):,} tokens")
        else:
            print(f"  [ATTENTION] Introuvable : corpus/humain/{theme}")
    full = "\n".join(by_theme.values())
    return full, by_theme


def load_ai_by_theme(base_dir, models, themes):
    by_theme = {t: [] for t in themes}
    flat_texts = []

    for model in models:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            print(f"  [ATTENTION] Introuvable : {model_dir}")
            continue
        subdirs = [d for d in os.listdir(model_dir)
                   if os.path.isdir(os.path.join(model_dir, d)) and d in themes]
        if subdirs:
            for theme in themes:
                text = load_txt_from_dir(os.path.join(model_dir, theme))
                if text:
                    by_theme[theme].append(text)
                    print(f"  [IA/{model}/{theme}] {len(text.split()):,} tokens")
        else:
            # Structure plate : on associe au thème selon le nom de fichier
            theme_found = False
            for fname in sorted(os.listdir(model_dir)):
                if not fname.endswith(".txt"):
                    continue
                fname_lower = fname.lower()
                for theme in themes:
                    if theme in fname_lower:
                        with open(os.path.join(model_dir, fname), encoding="utf-8") as f:
                            by_theme[theme].append(f.read())
                        theme_found = True
                        break
            if not theme_found:
                # Aucun thème détectable → texte plat global
                text = load_txt_from_dir(model_dir)
                if text:
                    flat_texts.append(text)
                    print(f"  [IA/{model}] {len(text.split()):,} tokens (sans thème)")

    # Construire les textes par thème
    by_theme_text = {}
    for theme in themes:
        parts = by_theme[theme]
        if parts:
            by_theme_text[theme] = "\n".join(parts)

    # Si des textes plats existent, les ajouter au global uniquement
    flat_global = "\n".join(flat_texts)
    full = "\n".join(list(by_theme_text.values()) + [flat_global])
    return full, by_theme_text


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()

# FRÉQUENCES PAR MILLION DE MOTS

def freq_per_million(tokens, term):
    if not tokens:
        return 0.0
    return (tokens.count(term) / len(tokens)) * 1_000_000


def category_freq_per_million(tokens, category):
    return sum(freq_per_million(tokens, t) for t in LEXICON[category])


def build_freq_table(corpora):
    """corpora = dict {nom: tokens_list}"""
    rows = []
    for name, tokens in corpora.items():
        row = {"corpus": name}
        for cat in LEXICON:
            row[cat] = category_freq_per_million(tokens, cat)
        rows.append(row)
    return pd.DataFrame(rows).set_index("corpus")


def build_theme_breakdown(human_by_theme, ai_by_theme):
    rows = []
    for theme in THEMES:
        for corpus_name, by_theme in [("Humain", human_by_theme), ("IA", ai_by_theme)]:
            text = by_theme.get(theme, "")
            tokens = tokenize(text) if text else []
            row = {"theme": THEME_LABELS[theme], "corpus": corpus_name}
            for cat in LEXICON:
                row[cat] = category_freq_per_million(tokens, cat)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index(["theme", "corpus"])
    return df

# COOCCURRENCE
def get_cooccurrences(tokens, target_terms, window=5):
    stop = set(nltk.corpus.stopwords.words("english"))
    cooc = Counter()
    for i, tok in enumerate(tokens):
        if tok in target_terms:
            start = max(0, i - window)
            end   = min(len(tokens), i + window + 1)
            context = [tokens[j] for j in range(start, end)
                       if j != i and tokens[j] not in stop and len(tokens[j]) > 2]
            cooc.update(context)
    return cooc


def cooccurrence_comparison(human_tokens, ai_tokens, target_terms, top_n=10):
    return {
        "Humain": get_cooccurrences(human_tokens, target_terms).most_common(top_n),
        "IA":     get_cooccurrences(ai_tokens,    target_terms).most_common(top_n),
    }

# VISUALISATIONS
os.makedirs("outputs", exist_ok=True)

CAT_COLORS = {
    "misogyne":     "#e63946",
    "neutre_genré": "#457b9d",
    "positif_genré":"#2a9d8f",
}


def plot_barplot_global(freq_df):
    """Barplot global humain vs IA."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(LEXICON))
    width = 0.35
    cats = list(LEXICON.keys())
    humain_vals = [freq_df.loc["humain", c] for c in cats]
    ia_vals     = [freq_df.loc["IA", c]     for c in cats]

    bars1 = ax.bar([i - width/2 for i in x], humain_vals, width,
                   label="Humain", color=[CAT_COLORS[c] for c in cats], alpha=0.9)
    bars2 = ax.bar([i + width/2 for i in x], ia_vals, width,
                   label="IA", color=[CAT_COLORS[c] for c in cats], alpha=0.45,
                   hatch="///", edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Fréquence / million de mots", fontsize=11)
    ax.set_title("Termes genrés — Humain vs IA (global)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/barplot_global.png", dpi=150)
    plt.close()
    print("[OK] outputs/barplot_global.png")


def plot_breakdown_by_theme(theme_df):
    themes_labels = list(THEME_LABELS.values())
    cats = list(LEXICON.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Breakdown par thème — termes genrés (/ 1M mots)",
                 fontsize=14, fontweight="bold", y=1.02)

    width = 0.35
    x = range(len(cats))

    for ax, theme_label in zip(axes, themes_labels):
        try:
            humain_vals = [theme_df.loc[(theme_label, "Humain"), c] for c in cats]
            ia_vals     = [theme_df.loc[(theme_label, "IA"),     c] for c in cats]
        except KeyError:
            ax.set_title(f"{theme_label}\n(données manquantes)", fontsize=11)
            continue

        ax.bar([i - width/2 for i in x], humain_vals, width,
               label="Humain", color=[CAT_COLORS[c] for c in cats], alpha=0.9)
        ax.bar([i + width/2 for i in x], ia_vals, width,
               label="IA", color=[CAT_COLORS[c] for c in cats], alpha=0.45,
               hatch="///", edgecolor="white")

        ax.set_title(theme_label, fontsize=12, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(["misogyne", "neutre", "positif"], fontsize=9)
        ax.set_ylabel("Fréquence / million de mots" if ax == axes[0] else "")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/breakdown_par_theme.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] outputs/breakdown_par_theme.png")


def plot_heatmap_misogyne_by_theme(theme_df):
    cats = list(LEXICON.keys())
    themes_labels = list(THEME_LABELS.values())
    data = {}
    for corpus in ["Humain", "IA"]:
        col = []
        for theme_label in themes_labels:
            try:
                col.append(theme_df.loc[(theme_label, corpus), "misogyne"])
            except KeyError:
                col.append(0.0)
        data[corpus] = col

    hm_df = pd.DataFrame(data, index=themes_labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(hm_df, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Fréquence misogyne / 1M mots"})
    ax.set_title("Intensité misogyne par thème", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/heatmap_misogyne_par_theme.png", dpi=150)
    plt.close()
    print("[OK] outputs/heatmap_misogyne_par_theme.png")


def plot_ratio_chart(theme_df):
    themes_labels = list(THEME_LABELS.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = range(len(themes_labels))

    for offset, corpus, color in [(-width/2, "Humain", "#e63946"),
                                   (+width/2, "IA",     "#457b9d")]:
        ratios = []
        for theme_label in themes_labels:
            try:
                miso = theme_df.loc[(theme_label, corpus), "misogyne"]
                posi = theme_df.loc[(theme_label, corpus), "positif_genré"]
                ratios.append(miso / posi if posi > 0 else 0)
            except KeyError:
                ratios.append(0)
        ax.bar([i + offset for i in x], ratios, width,
               label=corpus, color=color, alpha=0.85, edgecolor="white")

    ax.axhline(1, color="gray", linestyle="--", linewidth=1, label="ratio = 1 (équilibre)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(themes_labels, fontsize=11)
    ax.set_ylabel("Ratio misogyne / positif_genré", fontsize=11)
    ax.set_title("Rapport termes misogynes / positifs par thème", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/ratio_misogyne_positif.png", dpi=150)
    plt.close()
    print("[OK] outputs/ratio_misogyne_positif.png")


def plot_wordcloud(tokens, title, output_path, target_terms=None):
    freq = Counter(t for t in tokens if (target_terms is None or t in target_terms))
    if not freq:
        print(f"  [ATTENTION] Aucun terme pour : {title}")
        return
    wc = WordCloud(width=800, height=400, background_color="black",
                   colormap="plasma", max_words=60).generate_from_frequencies(freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=13, fontweight="bold", color="white",
              bbox=dict(facecolor="black", edgecolor="none", pad=4))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="black")
    plt.close()
    print(f"[OK] {output_path}")


def plot_cooccurrence_network(cooc_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"Humain": "#e63946", "IA": "#457b9d"}
    for ax, (corpus_name, top_words) in zip(axes, cooc_data.items()):
        color = colors[corpus_name]
        G = nx.Graph()
        G.add_node("[misogynes]", size=2000, color="#ffffff")
        for word, freq in top_words[:12]:
            G.add_node(word, size=300 + freq * 15, color=color)
            G.add_edge("[misogynes]", word, weight=freq)
        pos = nx.spring_layout(G, seed=42, k=1.5)
        nx.draw_networkx(G, pos, ax=ax,
                         node_size=[G.nodes[n].get("size", 300) for n in G.nodes],
                         node_color=[G.nodes[n].get("color", color) for n in G.nodes],
                         font_size=9, font_color="white", edge_color="#555555",
                         width=[G[u][v]["weight"] / 15 for u, v in G.edges],
                         with_labels=True)
        ax.set_title(f"Cooccurrences misogynes — {corpus_name}",
                     fontsize=11, fontweight="bold", color="white")
        ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()
    plt.savefig("outputs/reseau_cooccurrence.png", dpi=150, facecolor="#1a1a2e")
    plt.close()
    print("[OK] outputs/reseau_cooccurrence.png")

# RAPPORT
def generate_report(freq_df, theme_df, cooc_results):
    lines = [
        "=" * 65,
        "RAPPORT — ANALYSE LEXICALE MISOGYNE/GENRÉE",
        "Comparaison corpus humain vs corpus IA",
        "=" * 65,
        "",
        "--- FRÉQUENCES GLOBALES (/ million de mots) ---",
        freq_df.round(1).to_string(),
        "",
        "--- BREAKDOWN PAR THÈME ---",
        theme_df.round(1).to_string(),
        "",
        "--- RATIO MISOGYNE / POSITIF PAR THÈME ---"
    ]
    for theme_label in THEME_LABELS.values():
        lines.append(f"\n  {theme_label}")
        for corpus in ["Humain", "IA"]:
            try:
                miso = theme_df.loc[(theme_label, corpus), "misogyne"]
                posi = theme_df.loc[(theme_label, corpus), "positif_genré"]
                ratio = miso / posi if posi > 0 else float("inf")
                lines.append(f"    {corpus:<8} ratio = {ratio:.1f}  "
                              f"(misogyne={miso:.1f}, positif={posi:.1f})")
            except KeyError:
                lines.append(f"    {corpus} — données manquantes")

    lines += ["", "--- COOCCURRENTS DES TERMES MISOGYNES (top 10) ---"]
    for corpus_name, top_words in cooc_results.items():
        lines.append(f"\n  [{corpus_name}]")
        for word, count in top_words:
            lines.append(f"    {word:<20} {count}")

    lines += ["", "--- OBSERVATIONS AUTOMATIQUES ---"]
    if "humain" in freq_df.index and "IA" in freq_df.index:
        for cat in LEXICON:
            diff = freq_df.loc["IA", cat] - freq_df.loc["humain", cat]
            direction = "PLUS" if diff > 0 else "MOINS"
            lines.append(
                f"  Le corpus IA est {direction} '{cat}' que le corpus humain "
                f"({abs(diff):.1f}/M de différence)"
            )

    with open("outputs/rapport_lexical.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[OK] outputs/rapport_lexical.txt")

# PIPELINE PRINCIPAL
def main():
    print("=== Chargement du corpus humain ===")
    human_text, human_by_theme = load_human_by_theme(HUMAN_CORPUS_DIR, HUMAN_THEMES)

    print("\n=== Chargement du corpus IA (tous modèles fusionnés) ===")
    ai_text, ai_by_theme = load_ai_by_theme(AI_CORPUS_DIR, AI_MODELS, THEMES)

    # Fallback démo
    if not human_text:
        print("[DEMO] Corpus humain introuvable — texte de démo")
        human_text = (
            "she a bitch she a thot money cash bitch hoe slut girl shorty "
            "queen my girl she loyal bitch flex hoe ratchet thot girl woman shawty "
        ) * 300
        human_by_theme = {t: human_text for t in THEMES}

    if not ai_text:
        print("[DEMO] Corpus IA introuvable — texte de démo")
        ai_text = (
            "she a queen love her my girl girl woman bitch flex hoe thot "
            "bitch girl shorty lady love my girl queen goddess wife loyal "
        ) * 300
        ai_by_theme = {t: ai_text for t in THEMES}

    print("\n=== Tokenisation ===")
    human_tokens = tokenize(human_text)
    ai_tokens    = tokenize(ai_text)
    print(f"  Corpus humain : {len(human_tokens):,} tokens")
    print(f"  Corpus IA     : {len(ai_tokens):,} tokens")

    # Tokens par thème
    human_tokens_by_theme = {t: tokenize(txt) for t, txt in human_by_theme.items()}
    ai_tokens_by_theme    = {t: tokenize(txt) for t, txt in ai_by_theme.items()}

    print("\n=== Calcul des fréquences globales ===")
    freq_df = build_freq_table({"humain": human_tokens, "IA": ai_tokens})
    print(freq_df.round(1))

    print("\n=== Calcul du breakdown par thème ===")
    theme_df = build_theme_breakdown(human_by_theme, ai_by_theme)
    print(theme_df.round(1))

    print("\n=== Analyse de cooccurrence ===")
    cooc_results = cooccurrence_comparison(
        human_tokens, ai_tokens, LEXICON["misogyne"], top_n=10
    )

    print("\n=== Visualisations ===")
    plot_barplot_global(freq_df)
    plot_breakdown_by_theme(theme_df)
    plot_heatmap_misogyne_by_theme(theme_df)
    plot_ratio_chart(theme_df)

    all_terms = list(ALL_TERMS.keys())
    plot_wordcloud(human_tokens, "Termes genrés — Humain",
                   "outputs/wordcloud_humain.png", all_terms)
    plot_wordcloud(ai_tokens, "Termes genrés — IA",
                   "outputs/wordcloud_IA.png", all_terms)
    plot_cooccurrence_network(cooc_results)

    print("\n=== Rapport ===")
    generate_report(freq_df, theme_df, cooc_results)

    print("\n=== TERMINÉ — outputs/ ===")



# ANALYSE SPACY — CONTEXTE GRAMMATICAL DES TERMES GENRÉS
def spacy_context_analysis(human_text, ai_text, target_terms, n_examples=5):
    """
    Pour chaque terme cible, extrait :
      - sa fonction grammaticale dominante (sujet / objet / autre)
        via les dépendances spaCy
      - n_examples exemples de phrases réelles où il apparaît
    Sauvegarde un rapport texte + un barplot de distribution syntaxique.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"  [ATTENTION] spaCy indisponible ({e}) — section ignorée")
        return

    DEP_LABELS = {
        "nsubj": "sujet", "nsubjpass": "sujet passif",
        "dobj":  "objet direct", "pobj": "objet préposition",
        "attr":  "attribut", "appos": "apposition",
    }

    def analyse_corpus(text, corpus_name, target_terms):
        # spaCy sur des chunks de 100 000 caractères max (limite mémoire)
        results = {term: {"deps": Counter(), "examples": []} for term in target_terms}
        chunk_size = 80_000
        chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 400_000), chunk_size)]
        for chunk in chunks:
            doc = nlp(chunk)
            for token in doc:
                if token.text.lower() in target_terms:
                    term = token.text.lower()
                    dep = token.dep_
                    label = DEP_LABELS.get(dep, "autre")
                    results[term]["deps"][label] += 1
                    # Sauvegarder la phrase si pas trop longue
                    sent = token.sent.text.strip().replace("\n", " ")
                    if len(sent) < 200 and len(results[term]["examples"]) < n_examples:
                        results[term]["examples"].append(sent)
        return results

    print("  Analyse spaCy — corpus humain...")
    human_res = analyse_corpus(human_text, "Humain", target_terms)
    print("  Analyse spaCy — corpus IA...")
    ai_res    = analyse_corpus(ai_text,    "IA",     target_terms)

    lines = [
        "=" * 65,
        "ANALYSE SPACY — FONCTION SYNTAXIQUE DES TERMES MISOGYNES",
        "=" * 65
    ]
    for term in target_terms:
        h_deps = human_res[term]["deps"]
        a_deps = ai_res[term]["deps"]
        if not h_deps and not a_deps:
            continue
        lines += ["", f"TERME : « {term} »",
                  f"  Humain → {dict(h_deps.most_common(4))}",
                  f"  IA     → {dict(a_deps.most_common(4))}",
                  "  Exemples humain :"]
        for ex in human_res[term]["examples"][:3]:
            lines.append(f"    • {ex}")
        lines.append("  Exemples IA :")
        for ex in ai_res[term]["examples"][:3]:
            lines.append(f"    • {ex}")

    with open("outputs/rapport_spacy.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[OK] outputs/rapport_spacy.txt")

    # On prend les 5 termes misogynes les plus présents dans le corpus humain
    term_totals = {t: sum(human_res[t]["deps"].values()) for t in target_terms}
    top_terms = [t for t, _ in sorted(term_totals.items(), key=lambda x: -x[1])
                 if sum(human_res[t]["deps"].values()) > 0][:5]

    if not top_terms:
        return

    dep_cats = ["sujet", "objet direct", "objet préposition", "attribut", "autre"]
    fig, axes = plt.subplots(1, len(top_terms), figsize=(4 * len(top_terms), 5))
    if len(top_terms) == 1:
        axes = [axes]

    for ax, term in zip(axes, top_terms):
        h_vals = [human_res[term]["deps"].get(d, 0) for d in dep_cats]
        a_vals = [ai_res[term]["deps"].get(d, 0)    for d in dep_cats]
        x = range(len(dep_cats))
        w = 0.35
        ax.bar([i - w/2 for i in x], h_vals, w, label="Humain",
               color="#e63946", alpha=0.85)
        ax.bar([i + w/2 for i in x], a_vals, w, label="IA",
               color="#457b9d", alpha=0.85)
        ax.set_title(f'« {term} »', fontsize=11, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(dep_cats, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fonction syntaxique des termes misogynes — Humain vs IA",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/spacy_syntaxe.png", dpi=150)
    plt.close()
    print("[OK] outputs/spacy_syntaxe.png")


# BARPLOT COOCCURRENTS — comparaison visuelle des contextes
def plot_cooccurrence_barplot(cooc_results, top_n=15):
    """
    Barplot horizontal : top-15 cooccurrents pour Humain et IA,
    côte à côte pour comparer les contextes d'usage des termes misogynes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"Humain": "#e63946", "IA": "#457b9d"}

    for ax, (corpus_name, top_words) in zip(axes, cooc_results.items()):
        if not top_words:
            ax.set_title(f"{corpus_name} — aucune donnée")
            continue
        words  = [w for w, _ in top_words[:top_n]][::-1]
        counts = [c for _, c in top_words[:top_n]][::-1]
        ax.barh(words, counts, color=colors[corpus_name], alpha=0.85, edgecolor="white")
        ax.set_title(f"Contexte des termes misogynes — {corpus_name}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Nb de cooccurrences (fenêtre ±5 mots)")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Quels mots entourent les termes misogynes ?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/cooccurrence_barplot.png", dpi=150)
    plt.close()
    print("[OK] outputs/cooccurrence_barplot.png")

def main_v2():
    print("=== Chargement du corpus humain ===")
    human_text, human_by_theme = load_human_by_theme(HUMAN_CORPUS_DIR, HUMAN_THEMES)

    print("\n=== Chargement du corpus IA (tous modèles fusionnés) ===")
    ai_text, ai_by_theme = load_ai_by_theme(AI_CORPUS_DIR, AI_MODELS, THEMES)

    if not human_text:
        print("[DEMO] Corpus humain introuvable — texte de démo")
        human_text = (
            "she a bitch she a thot money cash bitch hoe slut girl shorty "
            "queen my girl she loyal bitch flex hoe ratchet thot girl woman shawty "
        ) * 300
        human_by_theme = {t: human_text for t in THEMES}

    if not ai_text:
        print("[DEMO] Corpus IA introuvable — texte de démo")
        ai_text = (
            "she a queen love her my girl girl woman bitch flex hoe thot "
            "bitch girl shorty lady love my girl queen goddess wife loyal "
        ) * 300
        ai_by_theme = {t: ai_text for t in THEMES}

    print("\n=== Tokenisation ===")
    human_tokens = tokenize(human_text)
    ai_tokens    = tokenize(ai_text)
    print(f"  Corpus humain : {len(human_tokens):,} tokens")
    print(f"  Corpus IA     : {len(ai_tokens):,} tokens")

    print("\n=== Calcul des fréquences globales ===")
    freq_df = build_freq_table({"humain": human_tokens, "IA": ai_tokens})
    print(freq_df.round(1))

    print("\n=== Calcul du breakdown par thème ===")
    theme_df = build_theme_breakdown(human_by_theme, ai_by_theme)
    print(theme_df.round(1))

    print("\n=== Analyse de cooccurrence ===")
    cooc_results = cooccurrence_comparison(
        human_tokens, ai_tokens, LEXICON["misogyne"], top_n=15
    )

    print("\n=== Analyse spaCy (fonction syntaxique) ===")
    spacy_context_analysis(human_text, ai_text, LEXICON["misogyne"], n_examples=5)

    print("\n=== Visualisations ===")
    plot_barplot_global(freq_df)
    plot_breakdown_by_theme(theme_df)
    plot_heatmap_misogyne_by_theme(theme_df)
    plot_ratio_chart(theme_df)

    all_terms = list(ALL_TERMS.keys())
    plot_wordcloud(human_tokens, "Termes genrés — Humain",
                   "outputs/wordcloud_humain.png", all_terms)
    plot_wordcloud(ai_tokens, "Termes genrés — IA",
                   "outputs/wordcloud_IA.png", all_terms)

    plot_cooccurrence_network(cooc_results)
    plot_cooccurrence_barplot(cooc_results, top_n=15)

    print("\n=== Rapport ===")
    generate_report(freq_df, theme_df, cooc_results)

    print("\n=== TERMINÉ — outputs/ ===")


if __name__ == "__main__":
    main_v2()
