import typing
from utils.sentiment_detection import read_tokens, load_reviews

#

def read_lexicon(filename: str) -> typing.Dict[str, int]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    """
    lexicon = {}
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(" ")
            for i in range(3):
                data[i] = data[i].split("=")[1]
            lexicon[data[0]] = 1 if data[2] == "positive\n" else -1
            #lexicon[data[0]] *= 1 if data[1] == "strong" else 1
    return lexicon


def predict_sentiment(tokens: typing.List[str], lexicon: typing.Dict[str, int]) -> int:
    """
    Given a list of tokens from a tokenized review and a lexicon, determine whether the sentiment of each review in the test set is
    positive or negative based on whether there are more positive or negative words.

    @param tokens: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    score = 0
    for token in tokens:
        score += lexicon.get(token, 0)
    return 1 if score >= 0 else -1


def accuracy(pred: typing.List[int], true: typing.List[int]) -> float:
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



def predict_sentiment_improved(tokens: typing.List[str], lexicon: typing.Dict[str, int]) -> int:
    """
    Use the training data to improve your classifier, perhaps by choosing an offset for the classifier cutoff which
    works better than 0.

    @param tokens: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1, -1 for positive and negative sentiments, respectively).
    """
    score = 0
    for token in tokens:
        score += lexicon.get(token, 0)
    return 1 if score > 13 else -1


def main():
    """
    Check your code locally (from the root director 'mlrd') by calling:
    PYTHONPATH='.' python3.6 exercises/tick1.py
    """
    review_data = load_reviews('data/sentiment_detection/reviews')
    tokenized_data = [read_tokens(fn['filename']) for fn in review_data]

    lexicon = read_lexicon('data/sentiment_detection/sentiment_lexicon')

    pred1 = [predict_sentiment(t, lexicon) for t in tokenized_data]
    acc1 = accuracy(pred1, [x['sentiment'] for x in review_data])
    print(f"Your accuracy: {acc1}")

    pred2 = [predict_sentiment_improved(t, lexicon) for t in tokenized_data]
    acc2 = accuracy(pred2, [x['sentiment'] for x in review_data])
    print(f"Your improved accuracy: {acc2}")


if __name__ == '__main__':
    main()
