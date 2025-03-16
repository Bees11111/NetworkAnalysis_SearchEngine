import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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


def print_top5_cosine_similar_documents(
    message, tf_vectorizer, tfidf_vectorizer, X_tf, X_tfidf, data_text
):
    """
    This function takes in a message and computes the top 5 most similar documents
    using both Term Frequency (TF) and Term Frequency-Inverse Document Frequency (TFIDF).

    Parameters:
        message : a query
        tf_vectorizer : a fitted TfidfVectorizer for the TF model
        tfidf_vectorizer : a fitted TfidfVectorizer for the TFIDF model
        X_tf : the document-term matrix for the TF model
        X_tfidf : the document-term matrix for the TFIDF model
        data_text : a DataFrame containing the documents

    Retruns:
        Prints the top 5 most similar documents for both TF and TFIDF, along with their similarity scores.
    """
    # Transform the query into a list of individual words
    message = message.lower().split()

    # Create the pseudo-document using the input message (joined into a string)
    pseudo_document_tf = tf_vectorizer.transform([" ".join(message)])
    pseudo_document_tfidf = tfidf_vectorizer.transform([" ".join(message)])

    # Compute cosine similarity between the query and all documents in both TF and TFIDF models
    similarities_tf = cosine_similarity(X_tf, pseudo_document_tf)
    similarities_tfidf = cosine_similarity(X_tfidf, pseudo_document_tfidf)

    # Get the indices of the top 5 most similar documents (sorted by similarity score)
    top5_tf = np.argsort(similarities_tf.flatten())[::-1][:5]
    top5_tfidf = np.argsort(similarities_tfidf.flatten())[::-1][:5]

    # Display the results for TF
    print(f"TF    : {top5_tf} | {similarities_tf[top5_tf].flatten()}")
    print(
        data_text.iloc[top5_tf].drop(columns=["text", "abstract", "references"]),
        "\n\n\n\n",
    )

    # Display the results for TFIDF
    print(f"TFIDF : {top5_tfidf} | {similarities_tfidf[top5_tfidf].flatten()}")
    print(data_text.iloc[top5_tfidf].drop(columns=["text", "abstract", "references"]))


def print_top5_euclidian_similar_documents(
    message, tf_vectorizer, tfidf_vectorizer, X_tf, X_tfidf, data_text
):
    """
    This function takes in a message and computes the top 5 most similar documents
    using both Term Frequency (TF) and Term Frequency-Inverse Document Frequency (TFIDF).

    Parameters:
        message : a query
        tf_vectorizer : a fitted TfidfVectorizer for the TF model
        tfidf_vectorizer : a fitted TfidfVectorizer for the TFIDF model
        X_tf : the document-term matrix for the TF model
        X_tfidf : the document-term matrix for the TFIDF model
        data_text : a DataFrame containing the documents

    Retruns:
        Prints the top 5 most similar documents for both TF and TFIDF, along with their similarity scores.
    """
    # Transform the query into a list of individual words
    message = message.lower().split()

    # Create the pseudo-document using the input message (joined into a string)
    pseudo_document_tf = tf_vectorizer.transform([" ".join(message)])
    pseudo_document_tfidf = tfidf_vectorizer.transform([" ".join(message)])

    # Compute cosine similarity between the query and all documents in both TF and TFIDF models
    similarities_tf = euclidean_distances(X_tf, pseudo_document_tf)
    similarities_tfidf = euclidean_distances(X_tfidf, pseudo_document_tfidf)

    # Get the indices of the top 5 most similar documents (sorted by similarity score)
    top5_tf = np.argsort(similarities_tf.flatten())[::-1][:5]
    top5_tfidf = np.argsort(similarities_tfidf.flatten())[::-1][:5]

    # Display the results for TF
    print(f"TF    : {top5_tf} | {similarities_tf[top5_tf].flatten()}")
    print(
        data_text.iloc[top5_tf].drop(columns=["text", "abstract", "references"]),
        "\n\n\n\n",
    )

    # Display the results for TFIDF
    print(f"TFIDF : {top5_tfidf} | {similarities_tfidf[top5_tfidf].flatten()}")
    print(data_text.iloc[top5_tfidf].drop(columns=["text", "abstract", "references"]))
