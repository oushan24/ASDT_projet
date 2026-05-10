import pronouncing
import re
from collections import Counter

def get_rhyme_part(word):
    cleaned = re.sub(r'[^\w]', '', word.lower())
    phones = pronouncing.phones_for_word(cleaned)
    return pronouncing.rhyming_part(phones[0]) if phones else cleaned

def analyze_by_windows(text, window_size=4):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    all_schemes = []
    total_rhyme_density = 0
    window_count = 0

    for i in range(0, len(lines), window_size):
        chunk = lines[i:i + window_size]
        if len(chunk) < 2: continue
        
        window_count += 1
        mapping = {}
        next_letter = 0
        normalized = ""
        rhyme_parts = []
        
        for line in chunk:
            last_word = line.split()[-1]
            part = get_rhyme_part(last_word)
            rhyme_parts.append(part)
            
            # Construction du schéma (AABB, ABAB...)
            if part not in mapping:
                mapping[part] = chr(65 + next_letter)
                next_letter += 1
            normalized += mapping[part]
        
        all_schemes.append(normalized)
        
        # Calcul de la densité pour cette fenêtre
        # On compte combien de lignes ont un partenaire de rime dans le bloc
        matches = 0
        for idx, part in enumerate(rhyme_parts):
            if rhyme_parts.count(part) > 1:
                matches += 1
        
        window_density = (matches / len(chunk)) * 100
        total_rhyme_density += window_density

    avg_density = round(total_rhyme_density / window_count, 2) if window_count > 0 else 0
    return avg_density, all_schemes

# Charger les fichiers
paths_humain = ["/life/life_corpus.txt", "/love/love_corpus.txt", "/money/money_corpus.txt"]
humain_corpus = ""
for i in paths_humain:
    path = "../corpus/humain" + i
    with open(path, 'r', encoding='utf-8') as f:
        humain_corpus += f.read()

densite_moyenne, liste_schemas = analyze_by_windows(humain_corpus, window_size=4)

print("Corpus humain")
print(f"DENSITÉ MOYENNE DES RIMES : {densite_moyenne}%")
print("\nGRADATION DES SCHÉMAS (par fréquence) :")
counts = Counter(liste_schemas)
for scheme, freq in counts.most_common():
    print(f"  - {scheme} : {freq} fois")

paths_gemma = ["/gemma_hood.txt", "/gemma_love.txt", "/gemma_money.txt"]
gemma_corpus = ""
for i in paths_gemma:
    path = "../corpus/IA/gemma" + i
    with open(path, 'r', encoding='utf-8') as f:
        gemma_corpus += f.read()

densite_moyenne2, liste_schemas2 = analyze_by_windows(gemma_corpus, window_size=4)

print("Corpus gemma")
print(f"DENSITÉ MOYENNE DES RIMES : {densite_moyenne2}%")
print("\nGRADATION DES SCHÉMAS (par fréquence) :")
counts2 = Counter(liste_schemas2)
for scheme, freq in counts2.most_common():
    print(f"  - {scheme} : {freq} fois")

paths_mixtral = ["/mixtral_hood.txt", "/mixtral_love.txt", "/mixtral_money.txt"]
mixtral_corpus = ""
for i in paths_mixtral:
    path = "../corpus/IA/mixtral" + i
    with open(path, 'r', encoding='utf-8') as f:
        mixtral_corpus += f.read()

densite_moyenne3, liste_schemas3 = analyze_by_windows(mixtral_corpus, window_size=4)

print("Corpus mixtral")
print(f"DENSITÉ MOYENNE DES RIMES : {densite_moyenne3}%")
print("\nGRADATION DES SCHÉMAS (par fréquence) :")
counts3 = Counter(liste_schemas3)
for scheme, freq in counts3.most_common():
    print(f"  - {scheme} : {freq} fois")

paths_qwen = ["/qwen_life.txt", "/qwen_love.txt", "/qwen_money.txt"]
qwen_corpus = ""
for i in paths_qwen:
    path = "../corpus/IA/qwen" + i
    with open(path, 'r', encoding='utf-8') as f:
        qwen_corpus += f.read()

densite_moyenne4, liste_schemas4 = analyze_by_windows(qwen_corpus, window_size=4)

print("Corpus qwen")
print(f"DENSITÉ MOYENNE DES RIMES : {densite_moyenne4}%")
print("\nGRADATION DES SCHÉMAS (par fréquence) :")
counts4 = Counter(liste_schemas4)
for scheme, freq in counts4.most_common():
    print(f"  - {scheme} : {freq} fois")