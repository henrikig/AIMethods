from dataset_trec import DatasetTrec
from math import log


# P(label) e.g. P("spam")
def prob_label(label, labels):
    return labels.count(label) / len(labels)


# P(word|label)
def prob_word_given_label(word, label, messages, labels):
    num_of_label = 0
    num_of_words = 0
    for i in range(len(labels)):
        if labels[i] == label:
            num_of_label += 1
            for j in range(len(messages[i])):
                if messages[i][j] == word:
                    num_of_words += 1
    return num_of_words/num_of_label


# Helper function for retrieving dict of probabilities
def dict_of_probabilities(label, messages, labels):
    prob_dict = dict()
    for message in messages:
        for word in message:
            if word not in prob_dict:
                prob_dict[word] = prob_word_given_label(word, label, messages, labels)
    return prob_dict


# P(spam|word_1, word_2, ..., word_n) (uses logarithmic sum to correct for underflow)
def naive_bayes_classifier(message, label, word_probs):
    p_label = prob_label(label, labels)
    probability = 0 + log(p_label + 1)
    for word in message:
        if word in word_probs:
            probability += log(word_probs[word] + 1)

    return probability


# Determine whether mails are spam or ham
def classification(messages, labels, training_data_size=100):
    # Get dicts of probabilities for word given label from words in training data
    spam_probs = dict_of_probabilities("spam", messages[:training_data_size], labels[:training_data_size])
    ham_probs = dict_of_probabilities("ham", messages[:training_data_size], labels[:training_data_size])

    verification_data = messages[training_data_size:training_data_size+100]
    verification_labels = labels[training_data_size:training_data_size+100]


    correctly_classified = 0
    wrongly_classified = 0

    for i in range(len(verification_data)):
        spam_prob = naive_bayes_classifier(verification_data[i], "spam", spam_probs)
        ham_prob = naive_bayes_classifier(verification_data[i], "ham", ham_probs)

        if spam_prob > ham_prob:
            if verification_labels[i] == "spam":
                correctly_classified += 1
            else:
                wrongly_classified += 1
        else:
            if verification_labels[i] == "ham":
                correctly_classified += 1
            else:
                wrongly_classified += 1

    return correctly_classified, wrongly_classified


if __name__ == "__main__":
    # Load messages and associated labels
    trec07 = DatasetTrec.load()
    messages = trec07.data
    labels = trec07.target

    results = []
    results.append([100, classification(messages, labels)])
    results.append([500, classification(messages, labels, training_data_size=500)])
    results.append([1000, classification(messages, labels, training_data_size=1000)])
    results.append([int(len(messages)/2), classification(messages, labels, training_data_size=int(len(messages)/2))])

    fmt_table_entry = '{number: <10} {correct: <10} {wrong}'

    print(f'{"#": <5} {"Correct": <10} Wrong')
    print(30*"-")
    for result in results:
        print(f'{result[0]: <7} {str(result[1][0]) + " %": <10} {str(result[1][1]) + " %"}')

