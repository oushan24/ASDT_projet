import os
import lyricsgenius
import time
import re 
import sys
import urllib.request
import urllib.parse
import json

# !!! avant d'executer le script il faut faire une commande suivante dans un venv dans le terminal pour enregistrer le token
# export GENIUS_ACCESS_TOKEN=ici il faut mettre le token individuel

# Vérification des arguments
if len(sys.argv) < 3:
    print("Erreur ! Utilisation correcte : python script.py \"Nom de l'artiste\" \"Nom de l'album\"")
    sys.exit(1)

artist_name = sys.argv[1]
album_name = sys.argv[2]

# Récupération du jeton d'accès via la variable d'environnement
token = os.environ.get("GENIUS_ACCESS_TOKEN")

if not token:
    print("Erreur : Le jeton d'accès n'est pas défini.")
    print("Veuillez l'exporter en utilisant : export GENIUS_ACCESS_TOKEN='votre_token_ici'")
    sys.exit(1)

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

    text = re.sub(r'\(\s*\n+', '(', text)
    text = re.sub(r'\n+\s*\)', ')', text)
    
    def join_brackets(match):
        return match.group(0).replace('\n', ' ').replace('  ', ' ')
    
    text = re.sub(r'\(.*?\)', join_brackets, text, flags=re.DOTALL)
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\d*Embed$', '', text)
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    
    return text

def get_tracklist(artist, album):
    """Récupère la liste des pistes de l'album via l'API ouverte d'iTunes (ne nécessite pas de clés)"""
    print(f"Recherche de la liste des pistes de l'album '{album}' de {artist} via la base de données Apple...")
    
    # Formatage d'une requête sécurisée pour l'URL
    query = urllib.parse.quote(f"{artist} {album}")
    url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=50"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            
        tracks = []
        for item in data.get('results', []):
            col_name = item.get('collectionName', '').lower()
            art_name = item.get('artistName', '').lower()
            
            # Vérification de correspondance partielle (parfois Apple ajoute des mots comme "Deluxe")
            if album.lower() in col_name and artist.lower() in art_name:
                tracks.append(item['trackName'])
        
        # Suppression des doublons (par exemple, versions Clean/Explicit), en conservant l'ordre initial des pistes
        return list(dict.fromkeys(tracks))
        
    except Exception as e:
        print(f"Erreur lors de la récupération de la liste des pistes : {e}")
        return []

try:
    # 1. Récupération de la liste des chansons en contournant Genius
    tracklist = get_tracklist(artist_name, album_name)
    
    if not tracklist:
        print("Album introuvable ou il ne contient aucune piste. Vérifiez l'orthographe.")
        sys.exit(1)
        
    print(f"\nPistes trouvées : {len(tracklist)}. Début du téléchargement des paroles depuis Genius...\n")
    
    safe_artist = artist_name.lower().replace(" ", "_")
    safe_album = album_name.lower().replace(" ", "_")
    filename = f"{safe_artist}_{safe_album}_corpus.txt"

    # 2. Recherche des paroles pour chaque chanson trouvée individuellement
    with open(filename, "w", encoding="utf-8") as f:
        for track_title in tracklist:
            print(f"Requête pour : {track_title}...")
            
            # Utilisation de la recherche de chanson spécifique, qui fonctionne sans erreur dans la bibliothèque
            song = genius.search_song(track_title, artist_name)

            if song:
                # 1. Rejeter si la langue est définie par Genius et n'est pas l'anglais ('en')
                if hasattr(song, 'language') and song.language and song.language != 'en':
                    print(f"-> Ignoré (Langue non anglaise détectée : {song.language}) : {song.title}")
                    continue
                
                # 2. Rejeter si le titre contient des mots indiquant une traduction
                title_lower = song.title.lower()
                translation_keywords = [
                    "translation", 
                    "traduction", 
                    "перевод", 
                    "traducción", 
                    "překlad", 
                    "翻译", 
                    "中文翻譯", 
                    "Deutsche Übersetzung", 
                    "Tradução em Português", 
                    "Svensk Översättning"
                    ]
                if any(word in title_lower for word in translation_keywords):
                    print(f"-> Ignoré (Version traduite détectée) : {song.title}")
                    continue
                # ----------------------------------

                if hasattr(song, 'lyrics') and song.lyrics:
                    pure_lyrics = clean_lyrics(song.lyrics).lower()
                    f.write(f"--- {song.title} ---\n")
                    f.write(pure_lyrics + "\n\n")
                else:
                    print(f"-> Ignoré (Paroles introuvables) : {track_title}")
            else:
                print(f"-> Ignoré (Chanson introuvable sur Genius) : {track_title}")
                
            time.sleep(1)
            
    print(f"\nTerminé ! Les paroles ont été enregistrées dans le fichier '{filename}'")

except Exception as e:
    print