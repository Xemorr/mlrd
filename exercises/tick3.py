import math

from utils.sentiment_detection import clean_plot, best_fit, chart_plot
import random
from typing import List, Tuple, Callable
from utils.sentiment_detection import read_tokens
import os

should_sample = False
tokens = []



def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    result = best_fit(token_frequencies_log, token_frequencies)
    print(f"Slope of zipf: {result[0]}, Y-Intercept of zipf: {result[1]}")
    func = lambda x: result[0] * x + result[1]
    return func


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    tokens_to_count = {}
    for token in tokens:
        tokens_to_count[token] = tokens_to_count.setdefault(token, 0) + 1
    tokens_and_frequency = [(x, tokens_to_count[x]) for x in tokens_to_count.keys()]
    tokens_and_frequency = sorted(tokens_and_frequency, key=lambda x: x[1], reverse=True)
    return tokens_and_frequency


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    data = [(x, frequencies[x][1]) for x in range(min(10000, len(frequencies)))]
    clean_plot()
    chart_plot(data, "Frequency Ranks", "word ranks", "frequency")


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]):
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    my_words = {"genuine", "interesting", "annoying", "bland", "too", "unfortunately", "fun", "satisfying", "recommend",
                "good"}
    my_word_rank_to_frequency = {}
    for i in range(min(10000, len(frequencies))):
        if frequencies[i][0] in my_words:
            my_word_rank_to_frequency[i] = frequencies[i][1]
    my_word_ranks = [(x, my_word_rank_to_frequency[x]) for x in my_word_rank_to_frequency.keys()]
    print(my_word_ranks)
    chart_plot(my_word_ranks, "Frequency Ranks",
               "word ranks", "frequency")
    return my_word_ranks


def draw_zipf(frequencies: List[Tuple[str, int]]):
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    log_log = []
    for x in range(0, min(len(frequencies), 10000)):
        log_log.append((math.log(x + 1), math.log(frequencies[x][1])))
    LoBF = estimate_zipf(log_log, [(x, frequencies[x][1]) for x in range(min(10000, len(frequencies)))])
    clean_plot()
    chart_plot(log_log, "Log-Log graph of first 10000 word frequencies and ranks", "Logged Frequencies", "Rank")
    chart_plot([(math.log(x + 1), LoBF(math.log(x + 1))) for x in range(10000)], "Log-Log graph of first 10000 word frequencies and ranks", "Logged Frequencies", "Rank")
    return LoBF

def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    output = []

    words = 0
    unique_words = 0
    token_set = set()
    power = 1
    for word in tokens:
        words += 1
        if word not in token_set:
            token_set.add(word)
            unique_words += 1
        if words == power:
            output.append((unique_words, words))
            power *= 2
            if power > len(tokens):
                power = len(tokens)
    return output


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    clean_plot()
    logged_type_counts = [(math.log(x[0]), math.log(x[1])) for x in type_counts]
    chart_plot(logged_type_counts, "Heaps' Law", "The number of unique words logged", "Logs of the total words")
    pass


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dataset_path = os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews')
    global tokens
    for file_name in random.sample(os.listdir(dataset_path), 1000) if should_sample else os.listdir(dataset_path):
        tokens += read_tokens(os.path.join(dataset_path, file_name))
    frequencies = count_token_frequencies(
        os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))  # set back to reviews_large at the end
    print(frequencies)

    draw_frequency_ranks(frequencies)
    rank_to_frequency = draw_selected_words_ranks(frequencies)

    clean_plot()
    LoBF = draw_zipf(frequencies)

    print([LoBF(x[0]) - x[1] for x in rank_to_frequency])

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    print(tokens)
    draw_heap(tokens)


if __name__ == '__main__':
    main()
