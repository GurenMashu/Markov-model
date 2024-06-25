import numpy as np
import string
from sklearn.model_selection import train_test_split

#edgar allen is 0, robert is 1

#preprocessing the data, converting a line to its raw form and giving unique index value to each word
def preprocessing(data_set):
    text_lines = []
    author_labels = []
    index = {}
    id = 0
    for num, data in enumerate(data_set):
        with open(data, encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    line = line.translate(str.maketrans('', '', string.punctuation)).lower().strip()
                    text_lines.append(line)
                    author_labels.append(num)
                    words = line.split()
                    for word in words:
                        if word not in index:
                            index[word] = id
                            id += 1
    unknown_word = id # stores the value for unknown data
    return index, author_labels, text_lines, id ,unknown_word

# Converting the text list into respective numbers using the index dict
def to_index(text_lines, index):
    text_indexed = []
    for text in text_lines:
        words = text.split()
        temp = [index[word] for word in words]
        text_indexed.append(temp)
    return text_indexed

# Train the model
def training(text_indexed, author_labels, id):
    edgar_A = np.zeros((id, id))
    robert_A = np.zeros((id, id))
    edgar_pi = np.zeros(id)
    robert_pi = np.zeros(id)

    for text, label in zip(text_indexed, author_labels):# A and pi matrices are trained by adding 1 for every hit
        first = text[0]
        if label == 0:
            edgar_pi[first] += 1
        elif label == 1:
            robert_pi[first] += 1

        for i in range(1,len(text) - 1):
            term = text[i]
            next_term = text[i + 1]
            if label == 0:
                edgar_A[term][next_term] += 1
            elif label == 1:
                robert_A[term][next_term] += 1

    return edgar_pi, edgar_A, robert_pi, robert_A

# Normalization
def normalize_log(pi, A):
    smoothed_pi = pi + 1
    smoothed_A = A + 1

    normalized_pi = smoothed_pi / np.sum(smoothed_pi)
    normalized_A = smoothed_A / np.sum(smoothed_A, axis=1, keepdims=True)

    log_pi = np.log(normalized_pi)
    log_A = np.log(normalized_A)
    return log_pi, log_A

def predict(text, index, unknown_word, edgar_log_pi, edgar_log_A, robert_log_pi, robert_log_A):
    #for the pi matrices
    unknown_word = len(edgar_log_pi)-1#has to be introduced due to indexing problems

    edgar_final_prob = edgar_log_pi[index.get(text[0],unknown_word)]
    robert_final_prob = robert_log_pi[index.get(text[0],unknown_word)]

    #for the A matrix
    for i in range(len(text)-1):
        term = index.get(text[i],unknown_word)
        next_term = index.get(text[i+1],unknown_word)

        edgar_final_prob += edgar_log_A[term,next_term]
        robert_final_prob += robert_log_A[term,next_term]

    return 1 if edgar_final_prob<robert_final_prob else 0



def model_preprocessing():#final methods, we have to split training and data sets
    data_set = ["D:/edgar_poems.txt","D:/robert_frost.txt"]

    index, author_labels, text_lines, id, unknown_word = preprocessing(data_set)
    text_indexed = to_index(text_lines, index)
    return text_indexed, index, author_labels, text_lines, id, unknown_word

def model_training(text_indexed, author_labels, id):
    edgar_pi, edgar_A, robert_pi, robert_A = training(text_indexed, author_labels, id)
    edgar_log_pi, edgar_log_A = normalize_log(edgar_pi, edgar_A)
    robert_log_pi, robert_log_A = normalize_log(robert_pi, robert_A)
    return edgar_log_pi, edgar_log_A, robert_log_pi, robert_log_A


def main():
    text_indexed, index, author_labels, text_lines, id, unknown_word = model_preprocessing()
    text_indexed_train, text_indexed_test, author_labels_train, author_labels_test = train_test_split(text_indexed, author_labels, test_size=.2, random_state = 42)
    edgar_log_pi, edgar_log_A, robert_log_pi, robert_log_A = model_training(text_indexed_train, author_labels_train, id)#preprocess trainging and test data
    count = 0
    for text_test, label_test in zip(text_indexed_test, author_labels_test):
        predicted_label = predict(text_test, index, unknown_word, edgar_log_pi, edgar_log_A, robert_log_pi, robert_log_A)

        if predicted_label == label_test:
            count+=1
    accuracy = count/(len(author_labels_test))
    print(f'The model accuracy comes to {accuracy*100:.2f}%')

if __name__ == '__main__':
    main()
