import lyricsgenius
import time
import re 
import sys
import os

access_token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not access_token:
        print("Erreur : Le token GENIUS_ACCESS_TOKEN n'est pas défini.")
        print("Avez-vous oublié de faire 'export GENIUS_ACCESS_TOKEN=votre_token' ?")
        sys.exit(1)

genius = lyricsgenius.Genius(
    token, 
    timeout=20, 
    retries=5, 
    remove_section_headers=True, 
    skip_non_songs=True
)

def clean_lyrics(text):
    lines = text.split('\n')
    if len(lines) > 0:
        lines = lines[1:]
    
    clean_text = '\n'.join(lines)
    clean_text = re.sub(r'\d*Embed$', '', clean_text)
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text).strip()
    return clean_text

artist_name = sys.argv[1]

try:
    print(f"Downloading songs by {artist_name}...")
    artist = genius.search_artist(artist_name, max_songs=15, sort="popularity")
    
    filename = f"{artist_name.lower()}_corpus.txt"

    with open(filename, "w", encoding="utf-8") as f:
        for song in artist.songs:
            pure_lyrics = clean_lyrics(song.lyrics).lower()
            f.write(f"--- {song.title} ---\n")
            f.write(pure_lyrics + "\n\n")
            
    print(f"Done, lyrics saved in '{artist_name}_corpus.txt'")

except Exception as e:
    print(f"Error: {e}")