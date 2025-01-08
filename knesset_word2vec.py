import json
import random
import os
import numpy as np
from gensim.models import Word2Vec
import string
import sys

def prep_corpus(corpus_file):
    print("reading and tokenizing corpus...")
    lines = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                lines.append(obj['sentence_text'])
            except:
                pass

    tokenized_sentences = []
    for sentence in lines:
        words = []
        for w in sentence.split():
            clean_w = w.strip(string.punctuation)
            if clean_w and (not clean_w.isdigit()):
                words.append(clean_w)
        tokenized_sentences.append(words)
    print(f"done reading. total sentences: {len(tokenized_sentences)}")
    return tokenized_sentences

def train_model(tokenized_sentences):
    print("training word2vec model...")
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=50,
        window=5,
        min_count=1
    )
    print("model trained.")
    return model

def find_similar_words(model, test_words):
    print("finding similar words...")
    results = {}
    for w in test_words:
        if w in model.wv:
            similar = model.wv.most_similar(w, topn=5)
            results[w] = similar
        else:
            results[w] = f"'{w}' not in vocabulary"
    print("found similar words.")
    return results

def make_sentence_embeddings(corpus_file, model):
    print("creating sentence embeddings...")
    lines = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                lines.append(obj['sentence_text'])
            except:
                pass

    sentence_embs = []
    for sentence in lines:
        valid_words = [w for w in sentence.split() if w in model.wv]
        if len(valid_words) == 0:
            vec = np.zeros(model.vector_size)
        else:
            vectors = [model.wv[w] for w in valid_words]
            vec = np.mean(vectors, axis=0)
        sentence_embs.append((sentence, vec))
    print(f"created embeddings for {len(sentence_embs)} sentences.")
    return sentence_embs

def student_cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def find_most_similar_sentences(sentence_embs, model):
    print("finding most similar sentences...")
    valid_indices = []
    for i, (sent, vec) in enumerate(sentence_embs):
        count_valid = 0
        for word in sent.split():
            if word in model.wv:
                count_valid += 1
        if count_valid >= 4:
            valid_indices.append(i)

    if len(valid_indices) == 0:
        print("no valid sentences found.")
        return []

    if len(valid_indices) <= 10:
        chosen = valid_indices
    else:
        chosen = random.sample(valid_indices, 10)

    results = []
    for idx in chosen:
        sentA, vecA = sentence_embs[idx]
        best_score = -1
        best_sent = None

        for j, (sentB, vecB) in enumerate(sentence_embs):
            if j == idx:
                continue
            sim = student_cosine_similarity(vecA, vecB)
            if sim > best_score:
                best_score = sim
                best_sent = sentB

        results.append((sentA, best_sent, best_score))
    print("found similar sentences.")
    return results

def replace_red_words_sentences(model):
    print("replacing red words in sentences...")
    data = [
        {
            "sentence": "בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים.",
            "red_words": ["דקות", "דיון"]
        },
        {
            "sentence": "בתור יושבת ראש הוועדה, אני מוכנה להאריך את ההסכם באותם תנאים.",
            "red_words": ["וועדה", "אני", "ההסכם"]
        },
        {
            "sentence": "בוקר טוב, אני פותח את הישיבה.",
            "red_words": ["בוקר", "פותח"]
        },
        {
            "sentence": "שלום, אנחנו שמחים להודיע שחברינו היקר קיבל קידום.",
            "red_words": ["שלום", "שמחים", "יקר", "קידום"]
        },
        {
            "sentence": "אין מניעה להמשיך לעסוק ב נושא.",
            "red_word": ["מניעה"]
        }
    ]

    all_results = []
    idx = 1
    for item in data:
        sent = item["sentence"]
        red_words = item.get("red_words", item.get("red_word", []))
        replaced_info = []
        new_sent = sent

        for rw in red_words:
            if rw in model.wv:
                sims = model.wv.most_similar(rw, topn=1)
                if len(sims) > 0:
                    new_w = sims[0][0]
                    new_sent = new_sent.replace(rw, new_w, 1)
                    replaced_info.append((rw, new_w))
                else:
                    replaced_info.append((rw, "no suggestion"))
            else:
                replaced_info.append((rw, "not in vocab"))
        all_results.append((idx, sent, new_sent, replaced_info))
        idx += 1
    print("replaced red words.")
    return all_results

if __name__ == "__main__":
    corpus_file = sys.argv[1]
    output_folder = sys.argv[2]

    # train model
    tok_sents = prep_corpus(corpus_file)
    model = train_model(tok_sents)

    # save the model
    print("saving model...")
    model_path = os.path.join(output_folder, "knesset_word2vec.model")
    model.save(model_path)
    print(f"model saved to {model_path}")

    # find similar words
    test_words = ["ישראל", "גברת", "ממשלה", "חבר", "בוקר", "מים", "אסור", "רשות", "זכויות"]
    similar_wds = find_similar_words(model, test_words)
    sim_words_file = os.path.join(output_folder, "knesset_similar_words.txt")
    print(f"writing similar words to {sim_words_file}...")
    with open(sim_words_file, 'w', encoding='utf-8') as f:
        for w, sim_list in similar_wds.items():
            if isinstance(sim_list, str):
                f.write(f"{w}: {sim_list}\n")
            else:
                sim_str = ", ".join([f"({sw}, {sc:.4f})" for sw, sc in sim_list])
                f.write(f"{w}: {sim_str}\n")
    print("done writing similar words.")

    # sentence embeddings and similar sentences
    sentence_embs = make_sentence_embeddings(corpus_file, model)
    top_sims = find_most_similar_sentences(sentence_embs, model)
    sim_sents_file = os.path.join(output_folder, "knesset_similar_sentences.txt")
    print(f"writing similar sentences to {sim_sents_file}...")
    with open(sim_sents_file, 'w', encoding='utf-8') as f:
        for orig, best, score in top_sims:
            f.write(f"{orig}\n")
            f.write("most similar sentence:\n")
            f.write(f"{best}\n\n")
    print("done writing similar sentences.")

    # replace red words
    replaced = replace_red_words_sentences(model)
    red_file = os.path.join(output_folder, "red_words_sentences.txt")
    print(f"writing replaced sentences to {red_file}...")
    with open(red_file, 'w', encoding='utf-8') as f:
        for idx, old_s, new_s, rep_info in replaced:
            f.write(f"{idx}: {old_s}: {new_s}\n")
            replaced_str = ", ".join([f"({o}->{n})" for o, n in rep_info])
            f.write(f"replaced words: {replaced_str}\n\n")
    print("all tasks done.")
