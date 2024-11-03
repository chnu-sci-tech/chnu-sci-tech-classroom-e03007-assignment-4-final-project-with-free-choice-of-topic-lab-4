import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from _collections_abc import Iterable
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


def get_cleaned_text(text: str):
    # Видяляємо html символи
    symbols = ("&le", "&gt", "&lt", "&ge")
    for symbol in symbols:
        text = text.replace(symbol, " ")
    # Видаляємо інші символи
    text = re.sub("[^a-zA-Zа-яА-Я\s]", "", text)
    # Текст до нижнього регістру + обрізання пробілів на початку і кінці
    return text.lower().strip()


# Для розкриття вкладених масивів
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


df = pd.read_csv("steam_reviews.csv")
df = df.sample(frac=1, replace=True)  # Перемішуємо датасет для різноманітності
# Видаляємо рядки, де є пусті значення
df = df.dropna(subset=["review", "title"])
df = df.head(1000)  # Залишаємо 10000 рядків, бо більше комп починає лагати
df = df.reset_index(drop=True)
# Видаляємо непотрібні колонки
df = df.drop(columns=["date_posted", "funny", "helpful", "hour_played", "is_early_access_review"])

# Створюємо словник стоп-слів
df["review"] = df["review"].apply(lambda data: get_cleaned_text(data))
stop_words = set(stopwords.words("english"))

polarity_list = []
subjectivity_list = []

# Розділяємо тексти на категорії на основі рекомендацій
for category in df["recommendation"].unique():
    # Фільтруємо відгуки
    texts = df.loc[df["recommendation"] == category]["review"].values.tolist()
    # Обчислюємо субєктивність та polarity
    polarity_values = [TextBlob(text).sentiment.polarity for text in texts]
    subjectivity_values = [TextBlob(text).sentiment.subjectivity for text in texts]
    # Видаляємо стоп-слова, токенізуємо
    filtered_texts = [
        " ".join([word for word in word_tokenize(text.lower()) if word not in stop_words]) for text in texts
    ]
    # Обчислюємо середні занчення
    avg_polarity = sum(polarity_values) / len(polarity_values)
    avg_subjectivity = sum(subjectivity_values) / len(subjectivity_values)
    polarity_list.append(avg_polarity)
    subjectivity_list.append(avg_subjectivity)
# Будуємо графік polarity за категоріями
plt.bar(df["recommendation"].unique(), polarity_list)
plt.xlabel("Категорія")
plt.ylabel("Середнє значення polarity")
plt.show()

# Будуємо графік subjectivity за категоріями
plt.bar(df["recommendation"].unique(), subjectivity_list)
plt.xlabel("Категорія")
plt.ylabel("Середнє значення subjectivity")
plt.show()

corpus = df["review"].values.tolist()
# Розбиваємо текст на токени
nouns = []
for text in corpus:
    tokens = word_tokenize(text)
    # Визначаємо частини мови для кожного токена
    tagged_tokens = pos_tag(tokens)

    # Відбираємо тільки іменники
    noun = [word for word, pos in tagged_tokens if pos.startswith("N")]
    nouns.append(noun)

# Зводимо усе в один масив
result = list(flatten(nouns))

# Тематизуємо текст
dictionary = Dictionary([result])
corpus = [dictionary.doc2bow(result)]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)
for topic_num, topic_words in lda_model.show_topics(num_topics=10, num_words=5, formatted=False):
    print("Тема {}: {}".format(topic_num, [word for word, prob in topic_words]))
    word_topics = lda_model.get_topics()

# Візуалізуємо результат
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
# Є проблеми з відображенням у юпітері, тому зберігаємо як html
pyLDAvis.display(vis_data)
# https://github.com/bmabey/pyLDAvis/issues/162
pyLDAvis.save_html(vis_data, "lda.html")
