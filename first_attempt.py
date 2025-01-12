import json
import re
from gensim.models import Word2Vec
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the Knesset corpus
def load_corpus(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                sentences.append(entry.get('sentence_text', ''))  # Use 'sentence_text' field
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line.strip()}")
    return sentences

# Preprocess and tokenize sentences
def preprocess_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        # Remove non-word tokens (e.g., punctuation, numbers)
        cleaned_sentence = re.sub(r'[^א-תa-zA-Z ]', '', sentence)
        # Tokenize by splitting on spaces
        tokens = cleaned_sentence.split()
        if tokens:  # Avoid empty lists
            tokenized_sentences.append(tokens)
    return tokenized_sentences

# Replace red words using most_similar
def replace_red_words(model, sentences, red_words, output_file):
    replaced_sentences = []
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, (sentence, words_to_replace) in enumerate(zip(sentences, red_words), 1):
            new_sentence = sentence
            replaced_words = []

            for word in words_to_replace:
                try:
                    similar_words = model.wv.most_similar(positive=[word], topn=3)
                    for candidate, _ in similar_words:
                        if candidate != word:
                            new_sentence = new_sentence.replace(word, candidate, 1)
                            replaced_words.append((word, candidate))
                            break
                except KeyError:
                    replaced_words.append((word, "[word not in vocabulary]"))

            file.write(f"{i}: {sentence}: {new_sentence}\n")
            file.write(f"replaced words: {', '.join([f'({orig}:{new})' for orig, new in replaced_words])}\n\n")

# Main script
if __name__ == "__main__":
    corpus_file = "knesset_corpus.jsonl"

    # Step 1: Load and preprocess corpus
    raw_sentences = load_corpus(corpus_file)
    tokenized_sentences = preprocess_sentences(raw_sentences)

    # Debug: Ensure preprocessing worked
    if not tokenized_sentences:
        raise ValueError("Preprocessed sentences are empty. Check the corpus file or preprocessing.")

    # Step 2: Create Word2Vec model
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=50,  # Balanced vector size
        window=5,        # Context window size
        min_count=1,     # Include rare words
        workers=4        # Utilize multi-core processing
    )

    # Step 3: Define sentences and red words
    sentences = [
        "בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים .",
        "בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים .",
        "בוקר טוב , אני פותח את הישיבה .",
        "שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום .",
        "אין מניעה להמשיך לעסוק ב נושא ."
    ]

    red_words = [
        ["דקות", "דיון"],
        ["הוועדה", "אני", "ההסכם"],
        ["בוקר", "פותח"],
        ["שלום", "שמחים", "היקר", "קידום"],
        ["מניעה"]
    ]

    # Step 4: Replace red words and save results
    replace_red_words(model, sentences, red_words, "red_words_sentences.txt")

