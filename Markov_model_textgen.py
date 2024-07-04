import nltk
import numpy as np
import string
from nltk import ngrams, word_tokenize


def Markov_chain(dataset):
    #creates and trains the dataset
    pi = {}
    a1 = {}
    a2 = {}

    pi_prob = {}
    a1_prob = {}
    a2_prob = {}
    with open(dataset, encoding='utf-8') as file:#opening,making text uniform and normalising
        for line in file:
            if line.strip():
                line = line.translate(str.maketrans('', '', string.punctuation)).lower().strip()
            tokens = nltk.word_tokenize(line)

            n_grams_1 = list(ngrams(tokens, 1))
            n_grams_2 = list(ngrams(tokens, 2))
            n_grams_3 = list(ngrams(tokens, 3))

            for text in n_grams_1:#if case is called if a word is not present and then adds 1, otherwise if the said value
                                    #is present,then it adds 1 to existing term
                if text[0] not in pi:
                    pi[text[0]] = 0
                pi[text[0]] += 1

            for text in n_grams_2:
                if text not in a1:
                    a1[text] = 0
                a1[text] += 1

            for text in n_grams_3:
                if text not in a2:
                    a2[text] = 0
                a2[text] += 1

    total_pi = sum(pi.values())
    total_a1 = sum(a1.values())
    total_a2 = sum(a2.values())

    for terms in pi:#normalising the terms
        pi_prob[terms] = pi[terms] / total_pi
    for terms in a1:
        a1_prob[terms] = a1[terms] / total_a1
    for terms in a2:
        a2_prob[terms] = a2[terms] / total_a2
    return pi_prob, a1_prob, a2_prob


def predict(pi_prob, a1_prob, a2_prob, current_word=None, previous_word=None):
    if current_word is None:#for first case
        current_word = np.random.choice(list(pi_prob.keys()), p=list(pi_prob.values()))
        return None, current_word

    if current_word is not None and previous_word is None:#for second case ie predicting second word
        potential_words = {key: value for key, value in a1_prob.items() if key[0] == current_word}#finds a dict of potential
                                                                                                    # words
        if potential_words:
            previous_word = current_word
            keys = list(potential_words.keys())
            values = np.array(list(potential_words.values()))
            values /= values.sum()  #normalisation is needed as error was raised due to sum
                                    #of probabilities not being one
            current_word = np.random.choice([key[1] for key in keys], p=values)
        else:
            raise ValueError("No suitable next word found.")
    else:
        potential_words = {key: value for key, value in a2_prob.items() if key[:2] == (previous_word, current_word)}
        if potential_words:
            previous_word = current_word
            keys = list(potential_words.keys())
            values = np.array(list(potential_words.values()))
            values /= values.sum()  # Normalize to sum to 1
            current_word = np.random.choice([key[2] for key in keys], p=values)
        else:
            # Handle case where no suitable next word is found in bigrams
            potential_words = {key: value for key, value in a1_prob.items() if key[0] == current_word}
            if potential_words:
                previous_word = current_word
                keys = list(potential_words.keys())
                values = np.array(list(potential_words.values()))
                values /= values.sum()  # Normalize to sum to 1
                current_word = np.random.choice([key[1] for key in keys], p=values)
            else:
                raise ValueError("No suitable next word found.")
    return previous_word, current_word


dataset = "robert_frost.txt"#change the file location here


pi_prob, a1_prob, a2_prob = Markov_chain(dataset)
current_word = previous_word = None
i = 0
poem = []
while i < 4:#no of lines
    try:
        j=0
        line = []
        while j < 6:#max no of words
            previous_word = current_word = None
            previous_word, current_word = predict(pi_prob, a1_prob, a2_prob, current_word, previous_word)
            line.append(current_word)
            j+=1
        i += 1
        poem.append(line)
    except ValueError as e:
        print(f"Error: {e}")
        break


def listToString(s):#converts to list to display in poem terms
    lines_as_strings = [" ".join(line) for line in s]
    return "\n".join(lines_as_strings)


print(listToString(poem))
#include stdio.h boii
