import json
import random
import os
import numpy as np
from gensim.models import Word2Vec
import string
import sys

def prepare_corpus(corpus_file_path):
    print("reading and tokenizing the corpus...")
    sentences = []
    with open(corpus_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                sentences.append(data['sentence_text'])
            except:
                pass

    tokenized_sentences = []
    for sentence in sentences:
        words = []
        for word in sentence.split():
            cleaned_word = word.strip(string.punctuation)
            if cleaned_word and not cleaned_word.isdigit():
                words.append(cleaned_word)
        tokenized_sentences.append(words)
    print(f"finished reading. total sentences: {len(tokenized_sentences)}")
    return tokenized_sentences

def train_word2vec_model(tokenized_sentences):
    print("training the word2vec model...")
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=150,
        window=7,
        min_count=1
    )
    print("word2vec model trained.")
    return model

def get_similar_words(model, words_to_test):
    print("finding similar words...")
    similar_words_dict = {}
    for word in words_to_test:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            similar_words_dict[word] = similar
        else:
            similar_words_dict[word] = f"'{word}' not in vocabulary"
    print("similar words found.")
    return similar_words_dict

def create_sentence_embeddings(corpus_file_path, model):
    print("creating sentence embeddings...")
    sentences = []
    with open(corpus_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                sentences.append(data['sentence_text'])
            except:
                pass

    embeddings = []
    for sentence in sentences:
        valid_words = [word for word in sentence.split() if word in model.wv]
        if not valid_words:
            vector = np.zeros(model.vector_size)
        else:
            vectors = [model.wv[word] for word in valid_words]
            vector = np.mean(vectors, axis=0)
        embeddings.append((sentence, vector))
    print(f"created embeddings for {len(embeddings)} sentences.")
    return embeddings

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def find_similar_sentences(sentence_embeddings, model):
    print("looking for similar sentences...")
    valid_sentence_indices = []
    for index, (sentence, vector) in enumerate(sentence_embeddings):
        valid_word_count = sum(1 for word in sentence.split() if word in model.wv)
        if valid_word_count >= 4:
            valid_sentence_indices.append(index)

    if not valid_sentence_indices:
        print("no valid sentences to compare.")
        return []

    selected_indices = valid_sentence_indices if len(valid_sentence_indices) <= 10 else random.sample(valid_sentence_indices, 10)

    similar_sentences = []
    for idx in selected_indices:
        original_sentence, original_vector = sentence_embeddings[idx]
        best_score = -1
        best_sentence = None

        for j, (comp_sentence, comp_vector) in enumerate(sentence_embeddings):
            if j == idx:
                continue
            similarity = cosine_similarity(original_vector, comp_vector)
            if similarity > best_score:
                best_score = similarity
                best_sentence = comp_sentence

        similar_sentences.append((original_sentence, best_sentence, best_score))
    print("similar sentences identified.")
    return similar_sentences

def replace_target_words(model):
    print("replacing target words in sentences...")
    data = [
        {
            "sentence": "בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים.",
            "target_words": ["דקות", "דיון"]
        },
        {
            "sentence": "בתור יושבת ראש הוועדה, אני מוכנה להאריך את ההסכם באותם תנאים.",
            "target_words": ["וועדה", "אני", "ההסכם"]
        },
        {
            "sentence": "בוקר טוב, אני פותח את הישיבה.",
            "target_words": ["בוקר", "פותח"]
        },
        {
            "sentence": "שלום, אנחנו שמחים להודיע שחברינו היקר קיבל קידום.",
            "target_words": ["שלום", "שמחים", "יקר", "קידום"]
        },
        {
            "sentence": "אין מניעה להמשיך לעסוק ב נושא.",
            "target_words": ["מניעה"]
        }
    ]

    replacement_results = []
    sentence_id = 1
    for item in data:
        original_sentence = item["sentence"]
        target_words = item.get("target_words", [])
        replaced_info = []
        new_sentence = original_sentence

        for target_word in target_words:
            if target_word in model.wv:
                similar = model.wv.most_similar(target_word, topn=1)
                if similar:
                    new_word = similar[0][0]
                    new_sentence = new_sentence.replace(target_word, new_word, 1)
                    replaced_info.append((target_word, new_word))
                else:
                    replaced_info.append((target_word, "no suggestion"))
            else:
                replaced_info.append((target_word, "not in vocab"))
        replacement_results.append((sentence_id, original_sentence, new_sentence, replaced_info))
        sentence_id += 1
    print("target words replaced.")
    return replacement_results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python script.py <corpus_file> <output_folder>")
        sys.exit(1)

    corpus_file = sys.argv[1]
    output_dir = sys.argv[2]

    # make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get the script's directory to save the model there
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # prepare and train the model
    tokenized_sentences = prepare_corpus(corpus_file)
    word2vec_model = train_word2vec_model(tokenized_sentences)

    # save the model in the script's directory
    model_save_path = os.path.join(script_directory, "knesset_word2vec.model")
    word2vec_model.save(model_save_path)
    print(f"model saved to {model_save_path}")

    # find similar words and save to output folder
    words_to_check = ["ישראל", "גברת", "ממשלה", "חבר", "בוקר", "מים", "אסור", "רשות", "זכויות"]
    similar_words = get_similar_words(word2vec_model, words_to_check)
    similar_words_file = os.path.join(output_dir, "knesset_similar_words.txt")
    with open(similar_words_file, 'w', encoding='utf-8') as file:
        for word, sims in similar_words.items():
            if isinstance(sims, str):
                file.write(f"{word}: {sims}\n")
            else:
                sims_formatted = ", ".join([f"({sim_word}, {sim_score:.4f})" for sim_word, sim_score in sims])
                file.write(f"{word}: {sims_formatted}\n")
    print(f"similar words saved to {similar_words_file}")

    # create sentence embeddings, find similar sentences, and save to output folder
    sentence_embeddings = create_sentence_embeddings(corpus_file, word2vec_model)
    similar_sentences = find_similar_sentences(sentence_embeddings, word2vec_model)
    similar_sentences_file = os.path.join(output_dir, "knesset_similar_sentences.txt")
    with open(similar_sentences_file, 'w', encoding='utf-8') as file:
        for original, similar, score in similar_sentences:
            file.write(f"{original}\n")
            file.write("most similar sentence:\n")
            file.write(f"{similar}\n\n")
    print(f"similar sentences saved to {similar_sentences_file}")

    # replace target words and save to output folder
    replaced_sentences = replace_target_words(word2vec_model)
    replaced_words_file = os.path.join(output_dir, "red_words_sentences.txt")
    with open(replaced_words_file, 'w', encoding='utf-8') as file:
        for idx, old_sentence, new_sentence, replacements in replaced_sentences:
            file.write(f"{idx}: {old_sentence} => {new_sentence}\n")
            replacements_str = ", ".join([f"({old}->{new})" for old, new in replacements])
            file.write(f"replaced words: {replacements_str}\n\n")
    print(f"replaced sentences saved to {replaced_words_file}")

    print("all tasks completed.")
