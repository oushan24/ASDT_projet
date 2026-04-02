def check_for_duplicates(file_path1, file_path2):
    try:
        # Read lines from the first file into a set
        with open(file_path1, 'r', encoding='utf-8') as f1:
            # .strip().lower() ensures "Song Name" matches "song name "
            songs1 = {line.strip().lower() for line in f1 if line.strip()}

        # Read lines from the second file into a set
        with open(file_path2, 'r', encoding='utf-8') as f2:
            songs2 = {line.strip().lower() for line in f2 if line.strip()}

        # Find the intersection (items existing in both sets)
        duplicates = songs1.intersection(songs2)

        if duplicates:
            print(f"--- Found {len(duplicates)} duplicate(s) ---")
            for song in sorted(duplicates):
                # Using .title() to make it readable, though original case is lost
                print(f"Duplicate found: {song}")
        else:
            print("Success: No duplicates found between the two files.")

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Assuming your files are named list1.txt and list2.txt
check_for_duplicates('/home/tupikina/Documents/Cours/Plurital/M_Valette/ASDT_projet/corpus/humain/all songs/love_songs2.txt', '/home/tupikina/Documents/Cours/Plurital/M_Valette/ASDT_projet/corpus/humain/all songs/tough_life_songs.txt')