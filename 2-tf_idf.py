from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Old documents
documents = [
    """The first detailed genetic map of cancer in pet cats reveals striking similarities with human versions of the disease, possibly helping find new ways to treat cancers in both.
    Scientists analysed tumour DNA from almost 500 domestic cats, uncovering key genetic mutations linked with the condition.
    Cancer is one of the main causes of illness and death in cats, however, very little is known about how it develops.""",

    """An altercation with a nightclub bouncer in Wellington, a wasteful Ashes performance and a foolish attempt to hide the truth, for which he later had to come clean.
    Few England captains have toured with such oversized baggage.
    Thousands of England fans travelled to Australia for the Ashes.
    Thousands more set alarms night after night, day after day back home.
    Brook owed them a performance and boy did he repay them here.
    It was the most mature of knocks after the most immature of winters.
    It is curious that Brook, a man with a T20 World Cup winner's medal from 2022 and a Test triple century, could perhaps have been accused of not having delivered a match-winning knock on the very biggest stage.
    His highest score against Australia is 85 in 10 Tests.""",

    """Chip giant Nvidia has reported record annual revenue of $215.9bn despite a wave of investor scepticism about the massive amounts of money being spent on artificial intelligence technology.
    The firm also beat analyst forecasts as sales for the last three months of its financial year jumped by 73 percent compared to 12 months earlier.
    Computing demand is growing exponentially, boss Jensen Huang said.
    Our customers are racing to invest in AI compute, the factories powering the AI industrial revolution and their future growth.
    While providing chips for companies across the AI sector, Nvidia has also laid out plans in recent weeks to generate demand with new technologies of its own."""
]

# New document
china_text = """China has become one of the largest technology markets in the world.
Chinese companies are investing heavily in artificial intelligence, computing and semiconductor technologies.
Many technology firms in China are developing new AI systems and expanding their global influence."""

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english"
)

# Fit on basic documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Transform the new China text
china_vector = vectorizer.transform([china_text])

# Show vocabulary
feature_names = vectorizer.get_feature_names_out()
print("Vocabulary:")
print(feature_names)
print()

# Show TF-IDF matrix for basic documents
print("TF-IDF Matrix for basic documents:")
print(tfidf_matrix.toarray())
print()

# Show China TF-IDF vector
print("China TF-IDF Vector:")
print(china_vector.toarray())
print()

# Cosine similarity
similarities = cosine_similarity(china_vector, tfidf_matrix)

print("Cosine Similarity with the China document:")
for i, sim in enumerate(similarities[0]):
    print(f"China vs Document {i+1}: {round(sim, 4)}")

most_similar_index = np.argmax(similarities)
print(f"\nThe China document is closest to Document {most_similar_index + 1}.")