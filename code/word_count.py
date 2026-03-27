import sys 

def compter_mots(nom_fichier):
    total_mots = 0
    try:
        with open(nom_fichier, "r", encoding="utf-8") as f:
            for ligne in f:
                mots = ligne.split()
                total_mots += len(mots)
        
        print(f"Nombre total de mots : {total_mots}")
        
    except FileNotFoundError:
        print("Erreur : Le fichier est introuvable.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if len(sys.argv) > 1:
    corpus = sys.argv[1]
    compter_mots(corpus)
    
else:
    print("Usage : python script.py <chemin_du_fichier>")
