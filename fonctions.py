import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def word_occurrences(data_text, num_words=25, visualisation=False):
    """
    Can plot a bar graph of the most common word occurrences in the 'text' column of data_text,
    and returns a dictionary of all word counts.

    Parameters:
        data_text (pd.DataFrame): DataFrame containing the 'text' column.
        num_words (int): Number of most common words to plot. Default is 25.

    Returns:
        dict: Dictionary with words as keys and their counts as values.
    """
    # Concatenate all text data into a single string
    all_text = " ".join(data_text["text"])

    # Split the text into words
    words = all_text.split()

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Get the most common words and their counts for plotting
    most_common_words = word_counts.most_common(num_words)

    # Separate the words and their counts for plotting
    plot_words, counts = zip(*most_common_words)

    if visualisation:
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=list(plot_words), y=list(counts), palette="rainbow_r")
        plt.xlabel("Words")
        plt.ylabel("Counts")
        plt.title(f"Top {num_words} Most Common Words in Text Data")
        plt.xticks(rotation=45)

        # Add count labels on top of each bar
        for i, count in enumerate(counts):
            ax.text(i, count, str(count), ha="center", va="bottom")

        plt.show()

    return dict(word_counts)
