import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

nltk.download("punkt")

sample_text = "I told my computer I needed a break. It said no problem and froze immediately."

# Sentence segmentation
sentence_list = sent_tokenize(sample_text)

for index, s in enumerate(sentence_list):
    print(f"Sentence segmentation{index + 1}: {s}")

print()

# Word segmentation
for idx, sentence in enumerate(sentence_list):
    token_list = word_tokenize(sentence)
    print(f"Word segmentation{idx + 1}: {token_list}")