import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score
#0 for Edgar Allen Poe, 1 for Robert Frost


"""when trying out the model, make sure the path to the txt files are correct"""


def preprocessing(data_set):
    # preprocessing the data, converting a line to its raw form and giving unique index value to each word
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


def to_index(text_lines, index):
    # Converting the text list into respective numbers using the index dictionary
    text_indexed = []
    for text in text_lines:
        words = text.split()
        temp = [index[word] for word in words]
        text_indexed.append(temp)
    return text_indexed

# Training the model
def training(text_indexed, author_labels, id):#initialize the A and pi arrays
    edgar_A = np.zeros((id+1, id+1))
    robert_A = np.zeros((id+1, id+1))
    edgar_pi = np.zeros(id+1)
    robert_pi = np.zeros(id+1)

    for text, label in zip(text_indexed, author_labels):
        # A and pi matrices are trained by adding 1 for every hit
        first = text[0]#the pi matrix part
        if label == 0:
            edgar_pi[first] += 1
        elif label == 1:
            robert_pi[first] += 1

        for i in range(1,len(text) - 1):#the A matrix part
            term = text[i]
            next_term = text[i + 1]
            if label == 0:
                edgar_A[term][next_term] += 1
            elif label == 1:
                robert_A[term][next_term] += 1

    return edgar_pi, edgar_A, robert_pi, robert_A


def normalize_log(pi, A):# Normalization of data using add one
    # smoothing and then taking the log of the said matrices
    smoothed_pi = pi + 1
    smoothed_A = A + 1

    normalized_pi = smoothed_pi / np.sum(smoothed_pi)
    normalized_A = smoothed_A / np.sum(smoothed_A, axis=1, keepdims=True)

    log_pi = np.log(normalized_pi)
    log_A = np.log(normalized_A)
    return log_pi, log_A


def priors(author_labels_train):#the priors are computed to prevent inaccuracies due to assymmetric datasets
    edgar = sum(y==0 for y in author_labels_train)
    robert = sum(y == 1 for y in author_labels_train)
    total = len(author_labels_train)
    edgar_prior = edgar/total
    robert_prior = robert / total
    #taking log of priors as thats the format of the trained data
    edgar_log_prior = np.log(edgar_prior)
    robert_log_prior = np.log(robert_prior)
    return edgar_log_prior,robert_log_prior


def predict(text, index, unknown_word, edgar_log_pi, edgar_log_A, robert_log_pi, robert_log_A, author_labels_train):

    # takes the first log probability for each of the groups
    edgar_final_prob = edgar_log_pi[index.get(text[0],unknown_word)]
    robert_final_prob = robert_log_pi[index.get(text[0],unknown_word)]
    edgar_log_prior, robert_log_prior = priors(author_labels_train)
    edgar_final_prob += edgar_log_prior
    robert_final_prob += robert_log_prior

    # adds together the log values for each of the author for a given term, term+1 value pair
    for i in range(len(text)-1):
        term = index.get(text[i],unknown_word)
        next_term = index.get(text[i+1],unknown_word)

        edgar_final_prob += edgar_log_A[term,next_term]
        robert_final_prob += robert_log_A[term,next_term]

    return 0 if edgar_final_prob>robert_final_prob else 1



def model_preprocessing():#pre-processes the data


    data_set = ["edgar_allen_poe.txt","robert_frost.txt"]#change the directory location according to where you've
                                                        # the dataset

    index, author_labels, text_lines, id, unknown_word = preprocessing(data_set)
    text_indexed = to_index(text_lines, index)
    return text_indexed, index, author_labels, text_lines, id, unknown_word

def model_training(text_indexed, author_labels, id):#trains the model data
    edgar_pi, edgar_A, robert_pi, robert_A = training(text_indexed, author_labels, id)
    edgar_log_pi, edgar_log_A = normalize_log(edgar_pi, edgar_A)
    robert_log_pi, robert_log_A = normalize_log(robert_pi, robert_A)
    return edgar_log_pi, edgar_log_A, robert_log_pi, robert_log_A


def main():#main function putting it all together
    #calculating model performance using f1 scoring

    avg_precision = 0
    avg_f1_score = 0
    avg_recall = 0
    for i in range(100):
        text_indexed, index, author_labels, text_lines, id, unknown_word = model_preprocessing()
        #splitting training and test data
        text_indexed_train, text_indexed_test, author_labels_train, author_labels_test = train_test_split(text_indexed,
                                                                                                          author_labels,
                                                                                                          test_size=.2,
                                                                                                          random_state = i)
        edgar_log_pi, edgar_log_A, robert_log_pi, robert_log_A = model_training(text_indexed_train,
                                                                                author_labels_train,
                                                                                id)#preprocess training and test data

        predictions = []#stores the predictions made by the predict method
        for text_test in (text_indexed_test):
            predicted_label = predict(text_test, index, unknown_word, edgar_log_pi,
                                      edgar_log_A, robert_log_pi,
                                      robert_log_A, author_labels_train)
            predictions.append(predicted_label)
        #calculating recall, f1 and precision scores
        recall = recall_score(author_labels_test,predictions)
        f1 = f1_score(author_labels_test,predictions)
        precision = precision_score(author_labels_test,predictions)

        avg_recall += recall
        avg_f1_score += f1
        avg_precision += precision

    avg_recall /= 100
    avg_f1_score /= 100
    avg_precision /= 100


    print(f'The average recall is {avg_recall * 100:.2f}%')
    print(f'The average f1 score is {avg_f1_score * 100:.2f}%')
    print(f'The average precision is {avg_precision * 100:.2f}%')

if __name__ == '__main__':
    main()

"""The average recall is 100.00%
The average f1 score is 79.80%
The average precision is 66.42%"""

"""The recall value of 100% indicates the model is very capable at capturing true positives
however the lower value of precision indicates 33.58% of all positives are falsely matched

The model clearly needs improvement specifically in terms of reducing false positives, which can potentially be 
rectified by improving the training data as the false positives might be arising due to ambiguous text where both 
authors are equally likely

"""
