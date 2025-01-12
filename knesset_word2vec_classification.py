# import json
# import sys
# import os
# import random
# import numpy as np
# from collections import Counter
# from gensim.models import Word2Vec
#
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.metrics import classification_report
#
#
# def get_top_two_speakers(corpus_file):
#     """
#     Identifies the two most frequent speakers in the corpus.
#
#     Parameters:
#         corpus_file (str): Path to the JSON Lines corpus file.
#
#     Returns:
#         list: List of tuples containing speaker names and their counts.
#               Example: [('Speaker A', 1500), ('Speaker B', 1200)]
#     """
#     speaker_counts = Counter()
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 entry = json.loads(line.strip())
#                 speaker = entry.get("speaker_name", "").strip()
#                 if speaker:
#                     speaker_counts[speaker] += 1
#             except json.JSONDecodeError:
#                 continue
#     return speaker_counts.most_common(2)
#
#
# def load_sentences(corpus_file, speaker_name, name_variations):
#     """
#     Loads all sentences spoken by the specified speaker, accounting for name variations.
#
#     Parameters:
#         corpus_file (str): Path to the JSON Lines corpus file.
#         speaker_name (str): Canonical name of the speaker.
#         name_variations (list): List of name variations for the speaker.
#
#     Returns:
#         list: List of sentences spoken by the speaker.
#     """
#     sentences = []
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 entry = json.loads(line.strip())
#                 current_speaker = entry.get("speaker_name", "").strip()
#                 if current_speaker == speaker_name or current_speaker in name_variations:
#                     sentence = entry.get("sentence_text", "").strip()
#                     if sentence:
#                         sentences.append(sentence)
#             except json.JSONDecodeError:
#                 continue
#     return sentences
#
#
# def embed_sentences(sentences, model):
#     """
#     Converts a list of sentences into their average Word2Vec embeddings.
#
#     Parameters:
#         sentences (list): List of sentences (strings).
#         model (Word2Vec): Pre-trained Word2Vec model.
#
#     Returns:
#         np.ndarray: Array of sentence embeddings.
#     """
#     embeddings = []
#     for sentence in sentences:
#         words = [word for word in sentence.split() if word in model.wv]
#         if not words:
#             emb = np.zeros(model.vector_size)
#         else:
#             word_vectors = [model.wv[word] for word in words]
#             emb = np.mean(word_vectors, axis=0)
#         embeddings.append(emb)
#     return np.array(embeddings)
#
#
# def classify_with_knn(X, y, n_neighbors=114, metric='cosine'):
#     """
#     Trains a KNN classifier, performs 5-fold cross-validation, and generates a classification report.
#
#     Parameters:
#         X (np.ndarray): Feature matrix.
#         y (np.ndarray): Labels.
#         n_neighbors (int): Number of neighbors for KNN.
#         metric (str): Distance metric for KNN.
#
#     Returns:
#         None
#     """
#     # Initialize KNN classifier with specified parameters
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
#
#     # Perform 5-fold cross-validation
#     print("Performing 5-fold cross-validation...")
#     cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#     print(f"KNN 5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
#
#     # Split the data into training and testing sets (80/20 split)
#     print("\nSplitting data into training and testing sets (80/20)...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     # Train the KNN classifier
#     print("Training KNN classifier...")
#     knn.fit(X_train, y_train)
#
#     # Predict on the test set
#     print("Predicting on the test set...")
#     y_pred = knn.predict(X_test)
#
#     # Generate classification report
#     print("\nClassification Report:")
#     print(classification_report(
#         y_test,
#         y_pred,
#         target_names=["Speaker 1", "Speaker 2"]
#     ))
#
#
# def main():
#     # Check command-line arguments
#     if len(sys.argv) != 3:
#         print("Usage: python knn_classification.py <path/to/corpus.jsonl> <path/to/model>")
#         sys.exit(1)
#
#     corpus_file = sys.argv[1]
#     model_path = sys.argv[2]
#
#     # Validate corpus file
#     if not os.path.isfile(corpus_file):
#         print(f"Error: Corpus file '{corpus_file}' does not exist.")
#         sys.exit(1)
#
#     # Validate model file
#     if not os.path.isfile(model_path):
#         print(f"Error: Word2Vec model '{model_path}' not found.")
#         sys.exit(1)
#
#     # Load the pre-trained Word2Vec model
#     print(f"Loading Word2Vec model from '{model_path}'...")
#     w2v_model = Word2Vec.load(model_path)
#     print(f"Model loaded successfully. Vector size: {w2v_model.vector_size}\n")
#
#     # Identify the two most frequent speakers
#     top_speakers = get_top_two_speakers(corpus_file)
#     if len(top_speakers) < 2:
#         print("Error: Less than two speakers found in the corpus.")
#         sys.exit(1)
#
#     speaker1_name, speaker1_count = top_speakers[0]
#     speaker2_name, speaker2_count = top_speakers[1]
#
#     print(
#         f"Top 2 Speakers:\n1. {speaker1_name} ({speaker1_count} sentences)\n2. {speaker2_name} ({speaker2_count} sentences)\n")
#
#     # Define name variations for each speaker
#     # **Important:** Update these lists based on your dataset's name variations
#     speaker1_variations = [
#         "ר' ריבלין", "ראובן ריבלין", "רובי ריבלין"  # Example variations for Speaker 1
#     ]
#
#     speaker2_variations = [
#         "אברהם בורג"  # Example variations for Speaker 2
#         # Add more variations if applicable
#     ]
#
#     # Load sentences for each speaker, accounting for name variations
#     print(f"Loading sentences for '{speaker1_name}' and '{speaker2_name}' (including variations)...")
#     sentences_speaker1 = load_sentences(corpus_file, speaker1_name, speaker1_variations)
#     sentences_speaker2 = load_sentences(corpus_file, speaker2_name, speaker2_variations)
#
#     print(
#         f"\nNumber of sentences loaded:\n- {speaker1_name}: {len(sentences_speaker1)}\n- {speaker2_name}: {len(sentences_speaker2)}\n")
#
#     # Balance the dataset by downsampling to the smaller class size
#     min_count = min(len(sentences_speaker1), len(sentences_speaker2))
#     if len(sentences_speaker1) != len(sentences_speaker2):
#         print(f"Balancing classes by downsampling to {min_count} sentences each...")
#         random.seed(42)  # For reproducibility
#         sentences_speaker1 = random.sample(sentences_speaker1, min_count)
#         sentences_speaker2 = random.sample(sentences_speaker2, min_count)
#         print(
#             f"After downsampling:\n- {speaker1_name}: {len(sentences_speaker1)} sentences\n- {speaker2_name}: {len(sentences_speaker2)} sentences\n")
#     else:
#         print("Classes are already balanced.\n")
#
#     # Embed sentences using the pre-trained Word2Vec model
#     print("Embedding sentences using Word2Vec model...")
#     embeddings_speaker1 = embed_sentences(sentences_speaker1, w2v_model)
#     embeddings_speaker2 = embed_sentences(sentences_speaker2, w2v_model)
#     print("Sentence embedding completed.\n")
#
#     # Combine embeddings and create labels
#     X = np.vstack([embeddings_speaker1, embeddings_speaker2])
#     y = np.array([0] * len(embeddings_speaker1) + [1] * len(embeddings_speaker2))
#
#     print(f"Total samples for classification: {len(y)}\n")
#
#     # Perform KNN classification with 5-fold cross-validation and generate classification report
#     classify_with_knn(X, y, n_neighbors=144, metric='cosine')
#
#
# if __name__ == "__main__":
#     main()

import json
import sys
import os
import random
import numpy as np
from collections import Counter
from gensim.models import Word2Vec

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize


def get_top_two_speakers(corpus_file):
    """
    Identifies the two most frequent speakers in the corpus.

    Parameters:
        corpus_file (str): Path to the JSON Lines corpus file.

    Returns:
        list: List of tuples containing speaker names and their counts.
              Example: [('Speaker A', 1500), ('Speaker B', 1200)]
    """
    speaker_counts = Counter()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                speaker = entry.get("speaker_name", "").strip()
                if speaker:
                    speaker_counts[speaker] += 1
            except json.JSONDecodeError:
                continue
    return speaker_counts.most_common(2)


def load_sentences(corpus_file, speaker_name, name_variations):
    """
    Loads all sentences spoken by the specified speaker, accounting for name variations.

    Parameters:
        corpus_file (str): Path to the JSON Lines corpus file.
        speaker_name (str): Canonical name of the speaker.
        name_variations (list): List of name variations for the speaker.

    Returns:
        list: List of sentences spoken by the speaker.
    """
    sentences = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                current_speaker = entry.get("speaker_name", "").strip()
                if current_speaker == speaker_name or current_speaker in name_variations:
                    sentence = entry.get("sentence_text", "").strip()
                    if sentence:
                        sentences.append(sentence)
            except json.JSONDecodeError:
                continue
    return sentences


def embed_sentences(sentences, model):
    """
    Converts a list of sentences into their average Word2Vec embeddings.

    Parameters:
        sentences (list): List of sentences (strings).
        model (Word2Vec): Pre-trained Word2Vec model.

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    embeddings = []
    for sentence in sentences:
        words = [word for word in sentence.split() if word in model.wv]
        if not words:
            emb = np.zeros(model.vector_size)
        else:
            word_vectors = [model.wv[word] for word in words]
            emb = np.mean(word_vectors, axis=0)
        embeddings.append(emb)
    return np.array(embeddings)


def classify_with_knn(X, y, n_neighbors=14, metric='euclidean'):
    """
    Trains a KNN classifier, performs 5-fold cross-validation, and generates a classification report.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        n_neighbors (int): Number of neighbors for KNN.
        metric (str): Distance metric for KNN.

    Returns:
        dict: Dictionary containing CV accuracy and classification report.
    """
    results = {}
    print(f"\n--- KNN Parameters: n_neighbors={n_neighbors}, metric='{metric}' ---")

    # Normalize if using cosine metric
    if metric == 'cosine':
        X_normalized = normalize(X)
    else:
        X_normalized = X.copy()

    # Initialize KNN classifier with specified parameters
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    # Perform 5-fold cross-validation
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(knn, X_normalized, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    results['cv_accuracy_mean'] = cv_scores.mean()
    results['cv_accuracy_std'] = cv_scores.std()

    # Split the data into training and testing sets (80/20 split)
    print("Splitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the KNN classifier
    print("Training KNN classifier...")
    knn.fit(X_train, y_train)

    # Predict on the test set
    print("Predicting on the test set...")
    y_pred = knn.predict(X_test)

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Speaker 1", "Speaker 2"]
    )
    print("Classification Report:")
    print(report)
    results['classification_report'] = report

    return results


def main():
    # Check command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python knn_classification.py <path/to/corpus.jsonl> <path/to/model>")
        sys.exit(1)

    corpus_file = sys.argv[1]
    model_path = sys.argv[2]

    # Validate corpus file
    if not os.path.isfile(corpus_file):
        print(f"Error: Corpus file '{corpus_file}' does not exist.")
        sys.exit(1)

    # Validate model file
    if not os.path.isfile(model_path):
        print(f"Error: Word2Vec model '{model_path}' not found.")
        sys.exit(1)

    # Load the pre-trained Word2Vec model
    print(f"Loading Word2Vec model from '{model_path}'...")
    w2v_model = Word2Vec.load(model_path)
    print(f"Model loaded successfully. Vector size: {w2v_model.vector_size}\n")

    # Ensure vector size is 100
    if w2v_model.vector_size != 100:
        print(
            f"Warning: Expected vector size 100, but got {w2v_model.vector_size}. Continuing with vector size {w2v_model.vector_size}.\n")

    # Identify the two most frequent speakers
    top_speakers = get_top_two_speakers(corpus_file)
    if len(top_speakers) < 2:
        print("Error: Less than two speakers found in the corpus.")
        sys.exit(1)

    speaker1_name, speaker1_count = top_speakers[0]
    speaker2_name, speaker2_count = top_speakers[1]

    print(
        f"Top 2 Speakers:\n1. {speaker1_name} ({speaker1_count} sentences)\n2. {speaker2_name} ({speaker2_count} sentences)\n")

    # Define name variations for each speaker
    # **Important:** Update these lists based on your dataset's name variations
    speaker1_variations = [
        "ר' ריבלין", "ראובן ריבלין", "רובי ריבלין"  # Example variations for Speaker 1
    ]

    speaker2_variations = [
        "אברהם בורג"  # Example variations for Speaker 2
        # Add more variations if applicable
    ]

    # Load sentences for each speaker, accounting for name variations
    print(f"Loading sentences for '{speaker1_name}' and '{speaker2_name}' (including variations)...")
    sentences_speaker1 = load_sentences(corpus_file, speaker1_name, speaker1_variations)
    sentences_speaker2 = load_sentences(corpus_file, speaker2_name, speaker2_variations)

    print(
        f"\nNumber of sentences loaded:\n- {speaker1_name}: {len(sentences_speaker1)}\n- {speaker2_name}: {len(sentences_speaker2)}\n")

    # Balance the dataset by downsampling to the smaller class size
    min_count = min(len(sentences_speaker1), len(sentences_speaker2))
    if len(sentences_speaker1) != len(sentences_speaker2):
        print(f"Balancing classes by downsampling to {min_count} sentences each...")
        random.seed(42)  # For reproducibility
        sentences_speaker1 = random.sample(sentences_speaker1, min_count)
        sentences_speaker2 = random.sample(sentences_speaker2, min_count)
        print(
            f"After downsampling:\n- {speaker1_name}: {len(sentences_speaker1)} sentences\n- {speaker2_name}: {len(sentences_speaker2)} sentences\n")
    else:
        print("Classes are already balanced.\n")

    # Embed sentences using the pre-trained Word2Vec model
    print("Embedding sentences using Word2Vec model...")
    embeddings_speaker1 = embed_sentences(sentences_speaker1, w2v_model)
    embeddings_speaker2 = embed_sentences(sentences_speaker2, w2v_model)
    print("Sentence embedding completed.\n")

    # Combine embeddings and create labels
    X = np.vstack([embeddings_speaker1, embeddings_speaker2])
    y = np.array([0] * len(embeddings_speaker1) + [1] * len(embeddings_speaker2))

    print(f"Total samples for classification: {len(y)}\n")

    # Define parameter grids
    metrics = ['euclidean', 'cosine']
    n_neighbors_list = [1,2,14, 100, 1000]

    # Store results
    all_results = []

    # Iterate over all combinations and print results
    for metric in metrics:
        for n_neighbors in n_neighbors_list:
            result = classify_with_knn(X, y, n_neighbors=n_neighbors, metric=metric)
            all_results.append({
                'metric': metric,
                'n_neighbors': n_neighbors,
                'cv_mean_accuracy': result['cv_accuracy_mean'],
                'cv_std_accuracy': result['cv_accuracy_std'],
                'classification_report': result['classification_report']
            })

    # Determine the best parameter combination based on CV accuracy
    best_result = max(all_results, key=lambda x: x['cv_mean_accuracy'])

    print("\n=== Summary of All Parameter Combinations ===\n")
    for res in all_results:
        print(f"Metric: {res['metric']}, Neighbors: {res['n_neighbors']}")
        print(f"  CV Accuracy: {res['cv_mean_accuracy']:.4f} (+/- {res['cv_std_accuracy']:.4f})")
        print(f"  Classification Report:\n{res['classification_report']}\n")

    print("=== Best Parameter Combination ===\n")
    print(f"Metric: {best_result['metric']}, Neighbors: {best_result['n_neighbors']}")
    print(f"CV Accuracy: {best_result['cv_mean_accuracy']:.4f} (+/- {best_result['cv_std_accuracy']:.4f})")
    print(f"Classification Report:\n{best_result['classification_report']}\n")

    print("Classification process completed.")


if __name__ == "__main__":
    main()
