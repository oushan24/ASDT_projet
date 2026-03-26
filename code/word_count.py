import sys 

def word_count(file):
    count = 0
    with open(file, "r") as f:
        for word in f.read():
            count += 1

    print(count)

corpus = sys.argv[1]

word_count(corpus)
