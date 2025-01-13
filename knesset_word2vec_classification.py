#
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
#     # find top 2 speakers
#     speaker_counts = Counter()
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 obj = json.loads(line.strip())
#                 spk = obj.get("speaker_name", "").strip()
#                 if spk:
#                     speaker_counts[spk] += 1
#             except:
#                 pass
#     return speaker_counts.most_common(2)
#
# def load_sents(corpus_file, spk_name, variations):
#     # load sents for speaker
#     sents = []
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 obj = json.loads(line.strip())
#                 cspk = obj.get("speaker_name", "").strip()
#                 if cspk == spk_name or cspk in variations:
#                     txt = obj.get("sentence_text", "").strip()
#                     if txt:
#                         sents.append(txt)
#             except:
#                 pass
#     return sents
#
# def embed_sents(sents, w2v):
#     # average word2vec
#     vecs = []
#     for sent in sents:
#         words = [w for w in sent.split() if w in w2v.wv]
#         if not words:
#             vecs.append(np.zeros(w2v.vector_size))
#         else:
#             arr = [w2v.wv[w] for w in words]
#             vecs.append(np.mean(arr, axis=0))
#     return np.array(vecs)
#
# def classify_knn(x, y, n_neighbors=5, metric='euclidean', weights='distance'):
#     # no prints except returning data
#     if metric == 'cosine':
#         x_ = normalize(x)
#     else:
#         x_ = x.copy()
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     cv_scores = cross_val_score(knn, x_, y, cv=skf, scoring='accuracy')
#     cv_mean = cv_scores.mean()
#     cv_std = cv_scores.std()
#
#     x_train, x_test, y_train, y_test = train_test_split(
#         x_, y, test_size=0.2, random_state=42, stratify=y
#     )
#     knn.fit(x_train, y_train)
#     y_pred = knn.predict(x_test)
#     report = classification_report(y_test, y_pred, target_names=["speaker1", "speaker2"])
#     return cv_mean, cv_std, report
#
# def main():
#     if len(sys.argv) != 3:
#         print("usage: python knn_classification.py <corpus> <model>")
#         sys.exit(1)
#
#     corpus_file = sys.argv[1]
#     model_path = sys.argv[2]
#
#     if not os.path.isfile(corpus_file) or not os.path.isfile(model_path):
#         print("error: corpus or model file not found")
#         sys.exit(1)
#
#     w2v = Word2Vec.load(model_path)
#
#     top_spk = get_top_two_speakers(corpus_file)
#     if len(top_spk) < 2:
#         print("error: not enough speakers")
#         sys.exit(1)
#
#     spk1, _ = top_spk[0]
#     spk2, _ = top_spk[1]
#
#     spk1_vars = ["ר' ריבלין", "ראובן ריבלין", "רובי ריבלין"]
#     spk2_vars = ["אברהם בורג"]
#
#     sents1 = load_sents(corpus_file, spk1, spk1_vars)
#     sents2 = load_sents(corpus_file, spk2, spk2_vars)
#
#     count_min = min(len(sents1), len(sents2))
#     if len(sents1) != len(sents2):
#         random.seed(42)
#         sents1 = random.sample(sents1, count_min)
#         sents2 = random.sample(sents2, count_min)
#
#     x1 = embed_sents(sents1, w2v)
#     x2 = embed_sents(sents2, w2v)
#     x = np.vstack([x1, x2])
#     y = np.array([0]*len(x1) + [1]*len(x2))
#
#     metrics = ['euclidean', 'cosine', 'manhattan', 'minkowski', 'chebyshev']
#     neighbors_list = [2,3,4,5,6,7,10,20,50,100,200,500,1000]
#     w = 'distance'
#
#     # store best for each metric
#     best_for_metric = {}
#     for m in metrics:
#         best_for_metric[m] = {
#             'n_neighbors': None,
#             'cv_mean': 0.0,
#             'cv_std': 0.0,
#             'report': ''
#         }
#
#     # iteration
#     for m in metrics:
#         for nn in neighbors_list:
#             print(f"testing: metric={m}, neighbors={nn}")
#             cv_mean, cv_std, rep = classify_knn(x, y, n_neighbors=nn, metric=m, weights=w)
#             if cv_mean > best_for_metric[m]['cv_mean']:
#                 best_for_metric[m]['n_neighbors'] = nn
#                 best_for_metric[m]['cv_mean'] = cv_mean
#                 best_for_metric[m]['cv_std'] = cv_std
#                 best_for_metric[m]['report'] = rep
#
#     print("\nfinal best combos per metric:\n")
#     for m in metrics:
#         data = best_for_metric[m]
#         print(f"metric={m}, neighbors={data['n_neighbors']}, acc={data['cv_mean']:.4f}, std={data['cv_std']:.4f}")
#         print(f"report:\n{data['report']}")
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
from sklearn.feature_extraction.text import TfidfVectorizer
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
    # load sentences for speaker
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

def vectorize_tfidf(sents, max_features=5000):
    # tf-idf vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(sents).toarray()
    return X

def classify_knn(x, y, n_neighbors=5, metric='euclidean', weights='distance'):
    # train and evaluate knn
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
    report = classification_report(y_test, y_pred, target_names=["speaker1", "speaker2"], zero_division=0)
    return cv_mean, cv_std, report

def main():
    if len(sys.argv) != 3:
        print("usage: python knn_classification.py <corpus.jsonl> <model>")
        sys.exit(1)

    corpus_file = sys.argv[1]
    model_path = sys.argv[2]

    if not os.path.isfile(corpus_file) or not os.path.isfile(model_path):
        print("error: corpus or model file not found")
        sys.exit(1)

    # load word2vec model
    w2v = Word2Vec.load(model_path)

    # get top 2 speakers
    top_spk = get_top_two_speakers(corpus_file)
    if len(top_spk) < 2:
        print("error: not enough speakers")
        sys.exit(1)

    spk1, _ = top_spk[0]
    spk2, _ = top_spk[1]

    # define name variations
    spk1_vars = ["ר' ריבלין", "ראובן ריבלין", "רובי ריבלין"]
    spk2_vars = ["אברהם בורג"]

    # load sentences
    sents1 = load_sents(corpus_file, spk1, spk1_vars)
    sents2 = load_sents(corpus_file, spk2, spk2_vars)

    # balance classes
    count_min = min(len(sents1), len(sents2))
    if len(sents1) != len(sents2):
        random.seed(42)
        sents1 = random.sample(sents1, count_min)
        sents2 = random.sample(sents2, count_min)

    # prepare representations
    # word2vec
    x1_w2v = embed_sents(sents1, w2v)
    x2_w2v = embed_sents(sents2, w2v)
    X_w2v = np.vstack([x1_w2v, x2_w2v])
    y_w2v = np.array([0]*len(x1_w2v) + [1]*len(x2_w2v))

    # tf-idf
    combined_sents = sents1 + sents2
    X_tfidf = vectorize_tfidf(combined_sents, max_features=5000)
    y_tfidf = np.array([0]*len(sents1) + [1]*len(sents2))

    # define parameters
    metrics = ['euclidean', 'cosine', 'manhattan', 'minkowski', 'chebyshev']
    neighbors_list = [2,3,4,5,6,7,10,20,50,100,200,500,1000]
    weights = 'distance'

    representations = {
        'word2vec': (X_w2v, y_w2v),
        'tfidf': (X_tfidf, y_tfidf)
    }

    best_results = {
        'word2vec': {},
        'tfidf': {}
    }

    # iterate over representations
    for rep_name, (X, y) in representations.items():
        best_results[rep_name] = {}
        for m in metrics:
            best_results[rep_name][m] = {
                'n_neighbors': None,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'report': ''
            }
            for nn in neighbors_list:
                print(f"testing: representation={rep_name}, metric={m}, neighbors={nn}")
                cv_mean, cv_std, rep = classify_knn(X, y, n_neighbors=nn, metric=m, weights=weights)
                if cv_mean > best_results[rep_name][m]['cv_mean']:
                    best_results[rep_name][m]['n_neighbors'] = nn
                    best_results[rep_name][m]['cv_mean'] = cv_mean
                    best_results[rep_name][m]['cv_std'] = cv_std
                    best_results[rep_name][m]['report'] = rep

    # print best results
    print("\nfinal best combinations:\n")
    for rep in best_results:
        print(f"representation: {rep}")
        for m in best_results[rep]:
            data = best_results[rep][m]
            print(f"metric={m}, neighbors={data['n_neighbors']}, acc={data['cv_mean']:.4f}, std={data['cv_std']:.4f}")
            print(f"report:\n{data['report']}\n")

    print("done.")

if __name__ == "__main__":
    main()
