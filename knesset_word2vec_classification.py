# import json
# import sys
# import os
# import random
# import numpy as np
# from collections import Counter
# from gensim.models import Word2Vec
#
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import normalize
#
# def get_top_two_speakers(corpus_file):
#     """
#     Identifies the two most frequent speakers in the corpus.
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
# def load_sentences(corpus_file, speaker_name, name_variations):
#     """
#     Loads all sentences spoken by the specified speaker or any of its name variations.
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
# def embed_sentences(sentences, model):
#     """
#     Converts a list of sentences into their average Word2Vec embeddings.
#     """
#     embeddings = []
#     for sentence in sentences:
#         words = [w for w in sentence.split() if w in model.wv]
#         if not words:
#             emb = np.zeros(model.vector_size)
#         else:
#             emb = np.mean([model.wv[w] for w in words], axis=0)
#         embeddings.append(emb)
#     return np.array(embeddings)
#
# def classify_with_knn(X, y, n_neighbors=5, metric='euclidean', weights='distance'):
#     """
#     Trains a KNN classifier with 5-fold cross-validation, then does a final 80/20 split for classification.
#
#     Returns a dict with:
#     - 'cv_mean_accuracy': mean CV accuracy
#     - 'cv_std_accuracy': std of CV accuracy
#     - 'classification_report': final classification report on the 80/20 split
#     """
#     results = {}
#     print(f"\n--- KNN Parameters: n_neighbors={n_neighbors}, metric='{metric}', weights='{weights}' ---")
#
#     # Normalize if metric is cosine
#     if metric == 'cosine':
#         X_normalized = normalize(X)
#     else:
#         X_normalized = X.copy()
#
#     # KNN classifier
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
#
#     # 5-Fold Stratified CV
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     print("Performing 5-fold cross-validation...")
#     cv_scores = cross_val_score(knn, X_normalized, y, cv=skf, scoring='accuracy')
#     cv_mean = cv_scores.mean()
#     cv_std = cv_scores.std()
#     print(f"Cross-validation Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
#     results['cv_mean_accuracy'] = cv_mean
#     results['cv_std_accuracy'] = cv_std
#
#     # Final train/test split (80/20)
#     print("Splitting data into training and testing sets (80/20)...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_normalized, y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     print("Training KNN classifier...")
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#
#     # Classification report
#     report = classification_report(y_test, y_pred, target_names=["Speaker 1", "Speaker 2"])
#     print("Classification Report:")
#     print(report)
#     results['classification_report'] = report
#
#     return results
#
# def main():
#     if len(sys.argv) != 3:
#         print("Usage: python knn_classification.py <path/to/corpus.jsonl> <path/to/model>")
#         sys.exit(1)
#
#     corpus_file = sys.argv[1]
#     model_path = sys.argv[2]
#
#     # Validate inputs
#     if not os.path.isfile(corpus_file):
#         print(f"Error: corpus file '{corpus_file}' not found.")
#         sys.exit(1)
#     if not os.path.isfile(model_path):
#         print(f"Error: Word2Vec model '{model_path}' not found.")
#         sys.exit(1)
#
#     # Load Word2Vec model
#     print(f"Loading Word2Vec model from '{model_path}'...")
#     w2v_model = Word2Vec.load(model_path)
#     print(f"Model loaded. Vector size: {w2v_model.vector_size}\n")
#
#     # Identify two most frequent speakers
#     top_speakers = get_top_two_speakers(corpus_file)
#     if len(top_speakers) < 2:
#         print("Error: Less than two distinct speakers found.")
#         sys.exit(1)
#
#     speaker1_name, speaker1_count = top_speakers[0]
#     speaker2_name, speaker2_count = top_speakers[1]
#     print(f"Top Speakers:\n1. {speaker1_name} ({speaker1_count})\n2. {speaker2_name} ({speaker2_count})\n")
#
#     # Name variations (customize as needed)
#     speaker1_variations = ["ר' ריבלין", "ראובן ריבלין", "רובי ריבלין"]
#     speaker2_variations = ["אברהם בורג"]
#
#     # Load sentences
#     print(f"Loading sentences for '{speaker1_name}' & '{speaker2_name}'...\n")
#     sents_speaker1 = load_sentences(corpus_file, speaker1_name, speaker1_variations)
#     sents_speaker2 = load_sentences(corpus_file, speaker2_name, speaker2_variations)
#
#     # Balance classes by downsampling
#     min_count = min(len(sents_speaker1), len(sents_speaker2))
#     if len(sents_speaker1) != len(sents_speaker2):
#         random.seed(42)
#         sents_speaker1 = random.sample(sents_speaker1, min_count)
#         sents_speaker2 = random.sample(sents_speaker2, min_count)
#         print(f"Downsampling to {min_count} each...\n")
#
#     # Embed
#     print("Embedding sentences...")
#     X_speaker1 = embed_sentences(sents_speaker1, w2v_model)
#     X_speaker2 = embed_sentences(sents_speaker2, w2v_model)
#     X = np.vstack([X_speaker1, X_speaker2])
#     y = np.array([0]*len(X_speaker1) + [1]*len(X_speaker2))
#     print(f"Total samples: {len(y)}\n")
#
#     # KNN parameter choices
#     metrics = ['euclidean', 'cosine', 'manhattan', 'minkowski', 'chebyshev']
#     n_neighbors_list = [3, 5, 7, 10, 20, 50, 100, 200, 1000]
#     weights = 'distance'
#
#     # We'll store best results per metric
#     best_results_per_metric = {}
#     # Initialize with the lowest possible accuracy, so we can update if we find higher
#     for m in metrics:
#         best_results_per_metric[m] = {
#             'n_neighbors': None,
#             'cv_mean_accuracy': 0.0,
#             'cv_std_accuracy': 0.0,
#             'classification_report': ''
#         }
#
#     all_results = []
#
#     for metric in metrics:
#         for n_neighbors in n_neighbors_list:
#             res = classify_with_knn(X, y, n_neighbors=n_neighbors, metric=metric, weights=weights)
#             all_results.append({
#                 'metric': metric,
#                 'n_neighbors': n_neighbors,
#                 'cv_mean_accuracy': res['cv_mean_accuracy'],
#                 'cv_std_accuracy': res['cv_std_accuracy'],
#                 'classification_report': res['classification_report']
#             })
#
#             # Check if we found a better combination for this metric
#             if res['cv_mean_accuracy'] > best_results_per_metric[metric]['cv_mean_accuracy']:
#                 best_results_per_metric[metric] = {
#                     'n_neighbors': n_neighbors,
#                     'cv_mean_accuracy': res['cv_mean_accuracy'],
#                     'cv_std_accuracy': res['cv_std_accuracy'],
#                     'classification_report': res['classification_report']
#                 }
#
#     print("\n=== Final Summary of All Results ===\n")
#     for res in all_results:
#         print(f"Metric: {res['metric']}, Neighbors: {res['n_neighbors']}")
#         print(f"  CV Accuracy: {res['cv_mean_accuracy']:.4f} (+/- {res['cv_std_accuracy']:.4f})")
#         print(f"  Classification Report:\n{res['classification_report']}\n")
#
#     print("=== Best Parameter Combination for Each Metric ===\n")
#     for m in metrics:
#         best = best_results_per_metric[m]
#         print(f"** {m.upper()} **")
#         print(f"Neighbors: {best['n_neighbors']}")
#         print(f"CV Accuracy: {best['cv_mean_accuracy']:.4f} (+/- {best['cv_std_accuracy']:.4f})")
#         print(f"Classification Report:\n{best['classification_report']}\n")
#
#     print("Done.")
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
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

def get_top_two_speakers(corpus_file):
    # find top 2 speakers
    speaker_counts = Counter()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                spk = obj.get("speaker_name", "").strip()
                if spk:
                    speaker_counts[spk] += 1
            except:
                pass
    return speaker_counts.most_common(2)

def load_sents(corpus_file, spk_name, variations):
    # load sents for speaker
    sents = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                cspk = obj.get("speaker_name", "").strip()
                if cspk == spk_name or cspk in variations:
                    txt = obj.get("sentence_text", "").strip()
                    if txt:
                        sents.append(txt)
            except:
                pass
    return sents

def embed_sents(sents, w2v):
    # average word2vec
    vecs = []
    for sent in sents:
        words = [w for w in sent.split() if w in w2v.wv]
        if not words:
            vecs.append(np.zeros(w2v.vector_size))
        else:
            arr = [w2v.wv[w] for w in words]
            vecs.append(np.mean(arr, axis=0))
    return np.array(vecs)

def classify_knn(x, y, n_neighbors=5, metric='euclidean', weights='distance'):
    # no prints except returning data
    if metric == 'cosine':
        x_ = normalize(x)
    else:
        x_ = x.copy()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(knn, x_, y, cv=skf, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    x_train, x_test, y_train, y_test = train_test_split(
        x_, y, test_size=0.2, random_state=42, stratify=y
    )
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=["speaker1", "speaker2"])
    return cv_mean, cv_std, report

def main():
    if len(sys.argv) != 3:
        print("usage: python knn_classification.py <corpus> <model>")
        sys.exit(1)

    corpus_file = sys.argv[1]
    model_path = sys.argv[2]

    if not os.path.isfile(corpus_file) or not os.path.isfile(model_path):
        print("error: corpus or model file not found")
        sys.exit(1)

    w2v = Word2Vec.load(model_path)

    top_spk = get_top_two_speakers(corpus_file)
    if len(top_spk) < 2:
        print("error: not enough speakers")
        sys.exit(1)

    spk1, _ = top_spk[0]
    spk2, _ = top_spk[1]

    spk1_vars = ["ר' ריבלין", "ראובן ריבלין", "רובי ריבלין"]
    spk2_vars = ["אברהם בורג"]

    sents1 = load_sents(corpus_file, spk1, spk1_vars)
    sents2 = load_sents(corpus_file, spk2, spk2_vars)

    count_min = min(len(sents1), len(sents2))
    if len(sents1) != len(sents2):
        random.seed(42)
        sents1 = random.sample(sents1, count_min)
        sents2 = random.sample(sents2, count_min)

    x1 = embed_sents(sents1, w2v)
    x2 = embed_sents(sents2, w2v)
    x = np.vstack([x1, x2])
    y = np.array([0]*len(x1) + [1]*len(x2))

    metrics = ['euclidean', 'cosine', 'manhattan', 'minkowski', 'chebyshev']
    neighbors_list = [2,3,4,5,6,7,10,20,50,100,200,500,1000]
    w = 'distance'

    # store best for each metric
    best_for_metric = {}
    for m in metrics:
        best_for_metric[m] = {
            'n_neighbors': None,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'report': ''
        }

    # iteration
    for m in metrics:
        for nn in neighbors_list:
            print(f"testing: metric={m}, neighbors={nn}")
            cv_mean, cv_std, rep = classify_knn(x, y, n_neighbors=nn, metric=m, weights=w)
            if cv_mean > best_for_metric[m]['cv_mean']:
                best_for_metric[m]['n_neighbors'] = nn
                best_for_metric[m]['cv_mean'] = cv_mean
                best_for_metric[m]['cv_std'] = cv_std
                best_for_metric[m]['report'] = rep

    print("\nfinal best combos per metric:\n")
    for m in metrics:
        data = best_for_metric[m]
        print(f"metric={m}, neighbors={data['n_neighbors']}, acc={data['cv_mean']:.4f}, std={data['cv_std']:.4f}")
        print(f"report:\n{data['report']}")

if __name__ == "__main__":
    main()
