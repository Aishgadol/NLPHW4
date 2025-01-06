import json
import math
import random
import os
import sys
import collections
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import multiprocessing

total_cores = multiprocessing.cpu_count()
cores_to_use = max(1, total_cores - 2)
os.environ["LOKY_MAX_CPU_COUNT"] = str(cores_to_use)

random.seed(42)
np.random.seed(42)

def get_2_speakers(corpus_file):
    # with this function we found the 2 most frequent speakers.
    speaker_names = []
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                speaker_names.append(entry['speaker_name'])
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line.strip()}")
    speaker_counts = Counter(speaker_names)
    most_common_speakers = speaker_counts.most_common(2)

    return most_common_speakers

class Speaker:
    def __init__(self, name):
        self.speaker_name = name
        self.data = []
        self.vectorizer = TfidfVectorizer()
        self.feature_matrix = None
        self.additional_features = None

def update_vectorizer_shared(speakers):

    all_lines = []
    for speaker in speakers:
        all_lines.extend([entry['sentence_text'] for entry in speaker.data])

    # create a shared vectorizer and fit it on all lines
    shared_vectorizer = TfidfVectorizer()
    shared_vectorizer.fit(all_lines)

    # transform each speaker's data with the shared vectorizer
    for speaker in speakers:
        lines = [entry['sentence_text'] for entry in speaker.data]
        speaker.vectorizer = shared_vectorizer
        speaker.feature_matrix = shared_vectorizer.transform(lines)


        # create a custom vector
        additional_features = []
        for entry in speaker.data:
            num_tokens = len(entry['sentence_text'].split())
            kneset_number = entry.get('kneset_number', 0)  # Default to 0 if not provided
            if entry.get('protocol_type', 0) == "plenary":
                protocol_type = 1
            else:
                protocol_type = 0
            additional_features.append([num_tokens, kneset_number, protocol_type])

        additional_features = np.array(additional_features)
        speaker.additional_features = additional_features

def get_data(speaker1, speaker2, others, corpus_file):
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                speaker = entry.get("speaker_name")
                # serch for the name of the speaker with all the vatiations we found for it's name
                if speaker == speaker1.speaker_name or speaker == "ר' ריבלין" or speaker == "ראובן ריבלין" or speaker == "רובי ריבלין":
                    speaker1.data.append(entry)
                elif speaker == speaker2.speaker_name or speaker == "אברהם בורג":
                    speaker2.data.append(entry)
                else:
                    others.data.append(entry)
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line.strip()}")

# we used this function to detect all the variations names of our 2 speakers
def get_speakers_with_expression(corpus_file, expression):

    matching_speakers = set()
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                speaker = entry.get("speaker_name")
                if speaker and expression in speaker:
                    matching_speakers.add(speaker)
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line.strip()}")
    return matching_speakers

def down_sampling(speaker1, speaker2, others):
    # get the number of lines
    target_count = min(len(speaker1.data), len(speaker2.data), len(others.data))

    # random sample lines
    downsampled_speaker1 = random.sample(speaker1.data, min(target_count, len(speaker1.data)))
    speaker1.data = downsampled_speaker1
    downsampled_speaker2 = random.sample(speaker2.data, min(target_count, len(speaker2.data)))
    speaker2.data = downsampled_speaker2
    downsampled_others = random.sample(others.data, min(target_count, len(others.data)))
    others.data = downsampled_others

    return

def create_models(speaker1, speaker2, others):
    '''
    1st model: binary knn with TfidfVectorizer
    2nd model: binary LR with TfidfVectorizer
    3rd model: 3-class knn with TfidfVectorizer
    4th model: 3-class LR with TfidfVectorizer
    5th model: binary knn with the new vector
    6th model: binary LR with the new vector
    7th model: 3-class knn with the new vector
    8th model: 3-class LR with the new vector
    '''
    # set the feature matrices and labels
    X12 = np.vstack([speaker1.feature_matrix.toarray(), speaker2.feature_matrix.toarray()])
    X34 = np.vstack([speaker1.feature_matrix.toarray(), speaker2.feature_matrix.toarray(), others.feature_matrix.toarray()])
    X56 = np.vstack([speaker1.additional_features, speaker2.additional_features])
    X78 = np.vstack([speaker1.additional_features, speaker2.additional_features, others.additional_features])

    y12 = np.array([0] * speaker1.feature_matrix.shape[0] + [1] * speaker2.feature_matrix.shape[0])
    y34 = np.array([0] * speaker1.feature_matrix.shape[0] + [1] * speaker2.feature_matrix.shape[0] + [2] * others.feature_matrix.shape[0])
    y56 = np.array([0] * speaker1.additional_features.shape[0] + [1] * speaker2.additional_features.shape[0])
    y78 = np.array([0] * speaker1.additional_features.shape[0] + [1] * speaker2.additional_features.shape[0] + [2] * others.additional_features.shape[0])

    models =[]
    # create models
    model1 = KNeighborsClassifier(n_neighbors=14, metric='euclidean')
    model1.fit(X12, y12)
    models.append(model1)

    model2 = LogisticRegression(max_iter=1000)
    model2.fit(X12, y12)
    models.append(model2)

    model3 = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
    model3.fit(X34, y34)
    models.append(model3)

    model4 = LogisticRegression(max_iter=1000)
    model4.fit(X34, y34)
    models.append(model4)

    model5 = KNeighborsClassifier(n_neighbors=4, metric='cosine')
    model5.fit(X56, y56)
    models.append(model5)

    model6 = LogisticRegression(max_iter=1000)
    model6.fit(X56, y56)
    models.append(model6)

    model7 = KNeighborsClassifier(n_neighbors=13, metric='cosine')
    model7.fit(X78, y78)
    models.append(model7)

    model8 = LogisticRegression(max_iter=1000)
    model8.fit(X78, y78)
    models.append(model8)

    # # evaluate the models with Classification Report
    # for idx, (model, data, labels) in enumerate(zip(
    #         models,
    #         [X12, X12, X34, X34, X56, X56, X78, X78],
    #         [y12, y12, y34, y34, y56, y56, y78, y78]
    # )):
    #     # split the data
    #     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    #
    #     # train the model
    #     model.fit(X_train, y_train)
    #
    #     # predict on the test set
    #     y_pred = model.predict(X_test)
    #
    #     # Classification Report
    #     model_name = f"Model {idx + 1}"
    #     print(f"\nClassification Report for {model_name}:\n")
    #     print(classification_report(y_test, y_pred))

    # # Perform 5-fold cross-validation
    # scores1 = cross_val_score(model1, X12, y12, cv=5, scoring='accuracy')
    # print(f"cross val 2 knn TfidfVectorizer: {scores1.mean():.4f}")
    # scores2 = cross_val_score(model2, X12, y12, cv=5, scoring='accuracy')
    # print(f"cross val 2 LR TfidfVectorizer: {scores2.mean():.4f}")
    # scores3 = cross_val_score(model3, X34, y34, cv=5, scoring='accuracy')
    # print(f"cross val 3 knn TfidfVectorizer: {scores3.mean():.4f}")
    # scores4 = cross_val_score(model4, X34, y34, cv=5, scoring='accuracy')
    # print(f"cross val 3 LR TfidfVectorizer: {scores4.mean():.4f}")
    # scores5 = cross_val_score(model5, X56, y56, cv=5, scoring='accuracy')
    # print(f"cross val 2 knn new vector: {scores5.mean():.4f}")
    # scores6 = cross_val_score(model6, X56, y56, cv=5, scoring='accuracy')
    # print(f"cross val 2 LR new vector: {scores6.mean():.4f}")
    # scores7 = cross_val_score(model7, X78, y78, cv=5, scoring='accuracy')
    # print(f"cross val 3 knn new vector: {scores7.mean():.4f}")
    # scores8 = cross_val_score(model8, X78, y78, cv=5, scoring='accuracy')
    # print(f"cross val 3 LR new vector: {scores8.mean():.4f}")

    return model1, model2, model3, model4, model5, model6, model7, model8

def test_model(model, test_file, vectorizer, output_folder, output_filename="classification_results.txt"):
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as file:
        for line in file:
            test_data.append(line.strip())

    # create vectorizer for the test, using the same vectorizer the speakers have
    test_matrix = vectorizer.transform(test_data)
    predictions = model.predict(test_matrix)

    label_map = {0: "first", 1: "second", 2: "other"}

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    # write predictions to the file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for idx, pred in enumerate(predictions):
            output_file.write(f"{label_map.get(pred, 'unknown')}\n")

    return


if __name__ == '__main__':

    corpus_file = sys.argv[1]
    test_file = sys.argv[2]
    output_folder = sys.argv[3]

    # get the 2 most common speakers
    most_common_speakers = get_2_speakers(corpus_file)
    # create the 3 classes
    speaker1 = Speaker(most_common_speakers[0][0])
    speaker2 = Speaker(most_common_speakers[1][0])
    others = Speaker("others")

    # this function saves each sentence in it's speakers class
    get_data(speaker1, speaker2, others, corpus_file)
    # down sampling to the number of sentences speaker2 has
    down_sampling (speaker1, speaker2, others)
    # create the sentences vectors for each class
    update_vectorizer_shared([speaker1, speaker2, others])
    '''
    1st model: binary knn with TfidfVectorizer
    2nd model: binary LR with TfidfVectorizer
    3rd model: 3-class knn with TfidfVectorizer
    4th model: 3-class LR with TfidfVectorizer
    5th model: binary knn with the new vector
    6th model: binary LR with the new vector
    7th model: 3-class knn with the new vector
    8th model: 3-class LR with the new vector
    '''
    # create all 8 models
    model1, model2, model3, model4, model5, model6, model7, model8 = create_models(speaker1, speaker2, others)
    # test the best 3-class model
    test_model(model4, test_file, speaker1.vectorizer, output_folder)




