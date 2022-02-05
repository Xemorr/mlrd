import math
import os
from functools import reduce
from typing import List, Dict, Tuple
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
from exercises.tick2 import calculate_class_log_probabilities, calculate_smoothed_log_probabilities, \
    predict_sentiment_nbc
from utils.sentiment_detection import read_tokens, load_reviews, split_data


def read_lexicon_magnitude(filename: str) -> Dict[str, Tuple[int, str]]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    """
    lexicon = {}
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(" ")
            for i in range(3):
                data[i] = data[i].split("=")[1]
            lexicon[data[0]] = (1 if data[2] == "positive\n" else -1, data[1])
    return lexicon


def predict_sentiment_magnitude(tokens: List[str], lexicon: Dict[str, Tuple[int, str]]) -> int:
    """
    Modify the simple classifier from Tick1 to include the information about the magnitude of a sentiment. Given a list
    of tokens from a tokenized review and a lexicon containing both sentiment and magnitude of a word, determine whether
    the sentiment of each review in the test set is positive or negative based on whether there are more positive or
    negative words. A word with a strong intensity should be weighted *four* times as high for the evaluator.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    total_score = 0
    for token in tokens:
        lexicon_element = lexicon.get(token, (0, "strong"))
        score = lexicon_element[0]
        magnitude = lexicon_element[1]
        total_score += score * {"strong": 4, "weak": 1}.get(magnitude)
    return 1 if total_score >= 0 else -1


def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)


def comb(n, r):
    return factorial(n) // factorial(r) // factorial(n - r)


def sign_test(actual_sentiments: List[int], classification_a: List[int], classification_b: List[int]) -> float:
    """
    Implement the two-sided sign test algorithm to determine if one classifier is significantly better or worse than
    another. The sign for a result should be determined by which classifier is more correct and the ceiling of the least
    common sign total should be used to calculate the probability.

    @param actual_sentiments: list of correct sentiment for each review
    @param classification_a: list of sentiment prediction from classifier A
    @param classification_b: list of sentiment prediction from classifier B
    @return: p-value of the two-sided sign test.
    """
    plus = 0  # a_correct
    minus = 0  # b_correct
    null = 0  # both_correct
    for i in range(len(actual_sentiments)):
        a_is_correct = classification_a[i] == actual_sentiments[i]
        b_is_correct = classification_b[i] == actual_sentiments[i]
        if a_is_correct and b_is_correct or (not(a_is_correct) and not(b_is_correct)):
            null += 1
        elif b_is_correct:
            minus += 1
        elif a_is_correct:
            plus += 1
    n = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)
    q = 0.5
    result = 0
    for i in range(k + 1):
        result += comb(n, i) * math.pow(q, i) * math.pow(1 - q, n - i)
    result *= 2
    return result  # p-value


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)

    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    dev_tokenized_data = [read_tokens(fn['filename']) for fn in validation_data][0:50]
    validation_sentiments = [x['sentiment'] for x in validation_data][0:50]

    lexicon_magnitude = read_lexicon_magnitude(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_magnitude = []
    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_magnitude(review, lexicon_magnitude)
        preds_magnitude.append(pred)
        pred_simple = predict_sentiment(review, lexicon)
        preds_simple.append(pred_simple)

    acc_magnitude = accuracy(preds_magnitude, validation_sentiments)
    acc_simple = accuracy(preds_simple, validation_sentiments)

    print(f"Your accuracy using simple classifier: {acc_simple}")
    print(f"Your accuracy using magnitude classifier: {acc_magnitude}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)

    preds_nb = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_nb.append(pred)

    acc_nb = accuracy(preds_nb, validation_sentiments)
    print(f"Your accuracy using Naive Bayes classifier: {acc_nb}\n")

    p_value_magnitude_simple = sign_test(validation_sentiments, preds_simple, preds_magnitude)
    print(
        f"The p-value of the two-sided sign test for classifier_a \"{'classifier simple'}\" and classifier_b \"{'classifier magnitude'}\": {p_value_magnitude_simple}")

    p_value_magnitude_nb = sign_test(validation_sentiments, preds_nb, preds_magnitude)
    print(
        f"The p-value of the two-sided sign test for classifier_a \"{'classifier magnitude'}\" and classifier_b \"{'naive bayes classifier'}\": {p_value_magnitude_nb}")


if __name__ == '__main__':
    main()
