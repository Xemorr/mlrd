import os
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table
import math
from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy

should_sample = False

def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    number_of_positive = 0
    number_of_negative = 0
    for review in training_data:
        sentiment = review["sentiment"]
        if sentiment == 1:
            number_of_positive += 1
        elif sentiment == -1:
            number_of_negative += 1

    return {1: math.log(number_of_positive / len(training_data)), 0: math.log((len(training_data) - number_of_positive - number_of_negative) / len(training_data)),
            -1: math.log(number_of_negative / len(training_data))}


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    result: Dict[int, Dict[str, float]] = {1: {}, -1: {}, 0: {}}
    words_in_each_class: Dict[int, int] = {1: 0, -1: 0, 0: 0}
    for review in training_data:
        review_sentiment = review["sentiment"]
        for word in review["text"]:
            for sentiment in [1, 0, -1]:
                result[sentiment][word] = result[sentiment].setdefault(word, 1) # changed default value to 1 for smoothing effect
            result[review_sentiment][word] += 1
            words_in_each_class[review_sentiment] += 1
    vocab_cardinality = len(result[-1].keys() | result[1].keys() | result[0].keys())
    for sentiment in [-1, 0, 1]:
        for word in result[sentiment]:
            result[sentiment][word] /= (words_in_each_class[sentiment] + vocab_cardinality)
            result[sentiment][word] = math.log(result[sentiment][word])
    return result


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    correct = 0
    for i in range(len(pred)):
        correct += 1 if pred[i] == true[i] else 0
    return correct / len(pred)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    log_word_sum_positive = 0
    log_word_sum_neutral = 0
    log_word_sum_negative = 0
    for word in review:
        if word in log_probabilities[1]:
            log_word_sum_positive += log_probabilities[1][word]
        if word in log_probabilities[0]:
            log_word_sum_neutral += log_probabilities[0][word]
        if word in log_probabilities[-1]:
            log_word_sum_negative += log_probabilities[-1][word]
    positive_log = class_log_probabilities[1] + log_word_sum_positive
    negative_log = class_log_probabilities[-1] + log_word_sum_negative
    neutral_log = class_log_probabilities[0] + log_word_sum_neutral
    maximum = max(positive_log, neutral_log, negative_log)
    return 1 if maximum == positive_log else (0 if maximum == neutral_log else (-1 if maximum == negative_log else 0))


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    N = 4
    average_Pe = 0
    average_Pa = 0
    k = agreement_table[list(agreement_table.keys())[0]][1] + agreement_table[list(agreement_table.keys())[0]][-1]
    for class_index in [1, -1]:
        counter = 0
        for document_index in agreement_table:
            counter += agreement_table[document_index][class_index]
        coefficient = (1/(N * k))
        average_Pe += (coefficient * counter) ** 2
    for document_index in agreement_table:
        counter = 0
        for class_index in agreement_table[document_index]:
            counter += agreement_table[document_index][class_index]
        #average_Pe += (sum(agreement_table[document_index].values()) * coefficient) ** 2
        average_Pa += (1 / (k * (k - 1))) * \
                      sum([agreement_table[document_index][class_index] * (agreement_table[document_index][class_index] - 1) for class_index in agreement_table[document_index]])
    average_Pa /= N
    print(f"My average Pa: {average_Pa}")
    print(f"My average Pe: {average_Pe}")
    return (average_Pa - average_Pe) / (1 - average_Pe)
    pass


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    dict = {0: {1: 0, -1: 0}, 1: {1 : 0, -1 : 0}, 2: {1: 0, -1: 0}, 3: {1: 0, -1: 0}}
    for review in review_predictions:
        for i in range(4):
            entry = dict[i]
            entry[review[i]] += 1
    return dict


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2022.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
