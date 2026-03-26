from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize
import nltk

nltk.download("punkt")

# Read text file
with open("pg74.txt", encoding="utf-8") as f:
    raw_text = f.read()

# Split text into sentences, then tokenize each sentence
sentences = [word_tokenize(sentence) for sentence in sent_tokenize(raw_text)]

# Clean tokens: lowercase, keep only alphabetic words, remove very short words
clean_text = []
for sentence in sentences:
    words = [word.lower() for word in sentence if word.isalpha() and len(word) > 3]
    if words:
        clean_text.append(words)

# Train Word2Vec model
model = Word2Vec(
    sentences=clean_text,
    vector_size=50,
    window=20,
    min_count=1,
    seed=42,
    workers=1
)

# Find 10 words most similar to "travel"
target_word = "travel"

if target_word in model.wv.key_to_index:
    similar_words = model.wv.most_similar(target_word, topn=10)
    print(f"Top 10 words similar to '{target_word}':\n")
    for word, score in similar_words:
        print(f"{word}: {score:.4f}")
else:
    print(f"The word '{target_word}' is not in the vocabulary.")