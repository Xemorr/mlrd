from typing import List, Dict, Union
import os
import random
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities


def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    result = [[] for _ in range(n)]
    for i in range((len(training_data) // n) * n):
        result[i % n].append(training_data[i])
    return result


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    pos = []
    neg = []
    for review in training_data:
        if review["sentiment"] == 1:
            pos.append(review)
        else:
            neg.append(review)
    training_data = pos + neg
    result = [[] for _ in range(n)]
    for i in range((len(training_data) // n) * n):
        result[i % n].append(training_data[i])
    return result


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    accuracies = []
    for fold_to_skip in range(len(split_training_data)):
        data = split_training_data[: fold_to_skip] + split_training_data[fold_to_skip + 1:]
        # flatten the data
        flattened = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                flattened.append(data[i][j])
        # train
        log_probabilities = calculate_smoothed_log_probabilities(flattened)
        class_log_probabilities = calculate_class_log_probabilities(flattened)
        # testing
        to_test = split_training_data[fold_to_skip]
        accuracies.append(accuracy(
            [predict_sentiment_nbc(review["text"], log_probabilities, class_log_probabilities) for review in to_test],
            [review["sentiment"] for review in to_test]))
    return accuracies

def cross_validate_tick_1(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    accuracies = []
    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')
    for fold_to_skip in range(len(split_training_data)):
        data = split_training_data[: fold_to_skip] + split_training_data[fold_to_skip + 1:]
        # flatten the data
        flattened = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                flattened.append(data[i][j])
        to_test = split_training_data[fold_to_skip]
        accuracies.append(accuracy(
            [predict_sentiment(review["text"], lexicon) for review in to_test],
            [review["sentiment"] for review in to_test]))
    return accuracies


def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return sum(accuracies) / len(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    mean = cross_validation_accuracy(accuracies)
    sum = 0
    for an_accuracy in accuracies:
        sum += (an_accuracy - mean) ** 2
    return sum / len(accuracies)


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    my_confusion_matrix = [[0, 0], [0, 0]]
    for i in range(len(predicted_sentiments)):
        predicted = 0 if predicted_sentiments[i] == 1 else 1
        actual = 0 if actual_sentiments[i] == 1 else 1
        my_confusion_matrix[predicted][actual] += 1
    return my_confusion_matrix


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    accuracies = cross_validate_tick_1(folds)
    print(f"Tick 1 stratified accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Tick 1 stratified mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Tick 1 stratified accuracy variance: {variance}\n")

    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')
    [predict_sentiment(data, lexicon) for data in recent_review_data], [x['sentiment'] for x in review_data]
    accuracy_2016 = accuracy()
    print(f"Tick 1 2016 accuracies: {accuracy_2016}")
    mean_accuracy = cross_validation_accuracy(accuracy_2016)
    print(f"Tick 1 2016 mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracy_2016)
    print(f"Tick 1 2016 accuracy variance: {variance}\n")

if __name__ == '__main__':
    main()
