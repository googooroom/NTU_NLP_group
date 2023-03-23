        

file_path="check.txt"


sentences = []
labels = []
def readfile():
    with open(file_path, 'r') as f:
        for line in f:
            sentence, label = line.strip().split('\t')
            sentences.append(sentence)
            labels.append(int(label)se these ten)

readfile()
print(sentences, labels)