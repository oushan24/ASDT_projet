'''
avant d'executer le script il faut faire une commande suivante dans le terminal et dans un venv pour enregistrer le token
export GENIUS_ACCESS_TOKEN=ici il faut mettre le token individuel

Format attendu de la liste des chansons : Artiste - Nom de la chanson

Utilisation du script : python search_song.py <fichier_liste.txt> <fichier_sortie.txt>
'''

import os
import lyricsgenius
import time
import re 
import sys

# Vérification des arguments
if len(sys.argv) < 3:
    print("Erreur ! Utilisation correcte : python search_song.py <fichier_liste.txt> <fichier_sortie.txt>")
    print("Exemple : python search_song.py songs.txt my_corpus.txt")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Récupération du token depuis les variables d'environnement
token = os.environ.get("GENIUS_ACCESS_TOKEN")

if not token:
    print("Erreur : Le token d'accès n'est pas défini.")
    print("Veuillez l'exporter : export GENIUS_ACCESS_TOKEN='votre_token'")
    sys.exit(1)

# Initialisation du client Genius
genius = lyricsgenius.Genius(
    token, 
    timeout=20, 
    retries=5, 
    remove_section_headers=True, 
    skip_non_songs=True
)

def clean_lyrics(text):
    if not text:
        return ""

    # On divise en lignes et on supprime la première ligne si elle contient l'en-tête de Genius (par exemple "Title Lyrics")
    lines = text.split('\n')
    if len(lines) > 0 and 'Lyrics' in lines[0]:
        lines = lines[1:]
    
    text = '\n'.join(lines)
    
    # Traitement des parenthèses
    text = re.sub(r'\(\s*\n+', '(', text)
    text = re.sub(r'\n+\s*\)', ')', text)
    
    def join_brackets(match):
        return match.group(0).replace('\n', ' ').replace('  ', ' ')
    
    text = re.sub(r'\(.*?\)', join_brackets, text, flags=re.DOTALL)
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    
    # Suppression des résidus de Genius à la fin (par exemple, "123Embed")
    text = re.sub(r'\d*Embed$', '', text)
    
    # Suppression des lignes vides superflues
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    
    return text

translation_keywords = [
    "translation", "traduction", "перевод", "traducción", 
    "překlad", "翻译", "中文翻譯", "deutsche übersetzung", 
    "tradução em português", "svensk översättning"
]

try:
    # Lecture de la liste des chansons
    with open(input_file, 'r', encoding='utf-8') as f:
        # Lecture des lignes non vides
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"{len(lines)} pistes trouvées dans le fichier. Début de l'analyse...\n")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in lines:
            # Format attendu "Artiste - Nom de la chanson"
            if " - " not in line:
                print(f"-> Format incorrect ignoré (attendu 'Artiste - Chanson') : {line}")
                continue
                
            artist_name, song_title = line.split(" - ", 1)
            artist_name = artist_name.strip()
            song_title = song_title.strip()
            
            print(f"Recherche : {artist_name} - {song_title}...")
            
            song = genius.search_song(song_title, artist_name)

            if song:
                # 1. Vérification de la langue
                if hasattr(song, 'language') and song.language and song.language != 'en':
                    print(f"-> Ignoré (Langue non anglaise : {song.language}) : {song.title}")
                    continue
                
                # 2. Vérification des mots indiquant une traduction dans le titre
                title_lower = song.title.lower()
                if any(word in title_lower for word in translation_keywords):
                    print(f"-> Ignoré (Version traduite détectée) : {song.title}")
                    continue

                # 3. Écriture du texte
                if hasattr(song, 'lyrics') and song.lyrics:
                    pure_lyrics = clean_lyrics(song.lyrics).lower()
                    out_f.write(f"--- {artist_name} - {song.title} ---\n")
                    out_f.write(pure_lyrics + "\n\n")
                else:
                    print(f"-> Ignoré (Paroles introuvables) : {song_title}")
            else:
                print(f"-> Ignoré (Chanson introuvable sur Genius) : {song_title}")
                
            # Pause pour éviter d'être bloqué par l'API
            time.sleep(1)
            
    print(f"\nTerminé ! Les textes ont été sauvegardés avec succès dans le fichier '{output_file}'")

except FileNotFoundError:
    print(f"Erreur : Fichier '{input_file}' introuvable.")
except Exception as e:
    print(f"Une erreur inattendue s'est produite : {e}")