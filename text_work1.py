import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from _collections_abc import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")
# Створюємо функцію для очистки тексту від непотрібних символів за доп. регулярних виразів


def get_cleaned_text(text: str):
    # Видяляємо html символи
    symbols = ("&le", "&gt", "&lt", "&ge")
    for symbol in symbols:
        text = text.replace(symbol, " ")
    # Видаляємо інші символи
    text = re.sub("[^a-zA-Zа-яА-Я\s]", "", text)
    # Текст до нижнього регістру + обрізання пробілів на початку і кінці
    return text.lower().strip()


def get_tokenized_text(text: str):
    return nltk.word_tokenize(text)


def get_without_stopwords(tokens):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    filtered_tokens = [token for token in tokens if not token in stop_words]
    return filtered_tokens


def get_stem_tokens(tokens):
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def get_lemmatize_text(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


# Для розкриття вкладених масивів


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


df = pd.read_csv("steam_reviews.csv")
# Видаляємо непотрібні колонки
df = df.drop(columns=["date_posted", "funny", "helpful", "hour_played", "is_early_access_review"])
# Видаляємо рядки, де є пусті значення
df = df.dropna(subset=["review", "title"])
# Формуємо корпус тексту
corpus = df["review"].tolist()
# Очищуємо текст (нормалізація)
corpus = [get_cleaned_text(review) for review in corpus]
# Проводимо токенізацію
corpus = [get_tokenized_text(review) for review in corpus]
# Видаляємо стоп-слова
corpus = [get_without_stopwords(review) for review in corpus]
# Стеммінг
corpus = [get_stem_tokens(review) for review in corpus]
# Лематизація
corpus = [" ".join(review) for review in corpus]
corpus = [get_lemmatize_text(review) for review in corpus]
# print(corpus[:5])

# Видаленння нечасто вживаних слів робить функція, що відповідає за стоп-слова
# Створюємо новий датасет з препроцесінговим текстом
new_df = pd.DataFrame({"cleaned_review": corpus, "recommendation": df["recommendation"], "title": df["title"]})
print(new_df.head(5))
# Фільтруємо дані за рекомендацією
recommended_reviews = new_df[new_df["recommendation"] == "Recommended"]

# Об'єднуємо всі препроцесовані відгуки в один список
recommended_words = recommended_reviews["cleaned_review"].values.tolist()
recommended_words = [review.split() for review in recommended_words]
# Рахуємо кількість входжень слова
recommended_words = list(flatten(recommended_words))
# Підраховуємо кількість входжень кожного слова
word_counts = Counter(recommended_words)

# Вибираємо топ-20 найпопулярніших слів
top_words = word_counts.most_common(21)  # 21, бо 1 - пробіл
del top_words[0]
# Візуалізуємо результати

plt.figure(figsize=(12, 6))
plt.barh([word[0] for word in top_words], [word[1] for word in top_words])
plt.title("Найбільш вживані слова в рекомендованих")
plt.xlabel("Count")
plt.show()
# Найгірші - найкращі з кінця
worst_words = word_counts.most_common()[-21:]
# Візуалізуємо
plt.figure(figsize=(12, 6))
plt.barh([word[0] for word in worst_words], [word[1] for word in worst_words])
plt.title("Найменш вживані слова в рекомендованих")
plt.xlabel("К-сть")
plt.show()

not_recommended_reviews = new_df[new_df["recommendation"] != "Recommended"]
# Об'єднуємо всі препроцесовані відгуки в один список
not_recommended_words = not_recommended_reviews["cleaned_review"].values.tolist()
not_recommended_words = [review.split() for review in not_recommended_words]
# Рахуємо кількість входжень слова
not_recommended_words = list(flatten(not_recommended_words))
# Підраховуємо кількість входжень кожного слова
word_counts = Counter(not_recommended_words)

# Вибираємо топ-20 найпопулярніших слів
top_words = word_counts.most_common(21)  # 21, бо 1 - пробіл
del top_words[0]
# Візуалізуємо результати

plt.figure(figsize=(12, 6))
plt.barh([word[0] for word in top_words], [word[1] for word in top_words])
plt.title("Найбільш вживані слова в не рекомендованих")
plt.xlabel("Count")
plt.show()

# Найгірші - найкращі з кінця
worst_words = word_counts.most_common()[-21:]
# Візуалізуємо
plt.figure(figsize=(12, 6))
plt.barh([word[0] for word in worst_words], [word[1] for word in worst_words])
plt.title("Найменш вживані слова в не рекомендованих")
plt.xlabel("К-сть")
plt.show()


# ініціалізуємо TfidfVectorizer
vectorizer = TfidfVectorizer()

# створюємо зважену матрицю термінів
tf_idf_matrix = vectorizer.fit_transform(corpus)

# отримуємо список мішків слів
terms = vectorizer.get_feature_names_out()

# роздруковуємо мішків слів
print("Мішки слів:", terms[:10])

# Побудуємо кластеризацію на основі TF-IDF для встановлення подібності текстів
# Беремо 500, бо комп не справляється
tfidf_matrix = vectorizer.fit_transform(corpus[:500])
# Кластеризація методом K-Means
kmeans = KMeans(n_clusters=5, random_state=0).fit(tfidf_matrix)

# Зменшення розмірності для візуалізації
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(tfidf_matrix.toarray())

# Візуалізація кластерів
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.predict(tfidf_matrix), cmap="viridis")
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker="x", s=200, linewidths=3, color="r")
plt.show()
# Як ми бачимо, що наші дані мають деяку закономірність.

# Повторюємо попередні етапи для суміші n-грам та порівняти отримані результати
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(corpus[:500])
# Кластеризація методом K-Means
kmeans = KMeans(n_clusters=5, random_state=0).fit(tfidf_matrix)

# Зменшення розмірності для візуалізації
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(tfidf_matrix.toarray())

# Візуалізація кластерів
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.predict(tfidf_matrix), cmap="viridis")
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker="x", s=200, linewidths=3, color="r")
plt.show()
# Тут дані більш корелюють, та мають менший розкид
