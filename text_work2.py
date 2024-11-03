import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

df = pd.read_csv('steam_reviews.csv')
df = df.sample(frac=1, replace=True)  # Перемішуємо датасет для різноманітності
# Видаляємо рядки, де є пусті значення
df = df.dropna(subset=['review', 'title'])
df = df.head(1000)  # Залишаємо 10000 рядків, бо більше комп починає лагати
df = df.reset_index(drop=True)
# Видаляємо непотрібні колонки
df = df.drop(columns=['date_posted', 'funny', 'helpful',
             'hour_played', 'is_early_access_review'])

train_data, test_data = train_test_split(df, test_size=0.3)

# Відокремлюємо текст відгуку та значення рекоменадції
train_text = train_data["review"].values.astype("U")
train_labels = train_data["recommendation"].values
test_text = test_data["review"].values.astype("U")
test_labels = test_data["recommendation"].values

vectorizer = CountVectorizer(stop_words="english", lowercase=True)
train_vectors = vectorizer.fit_transform(train_text)
test_vectors = vectorizer.transform(test_text)

# Застосувуємо алгоритм логістичної регресії
clf = LogisticRegression()
clf.fit(train_vectors, train_labels)

# Метрики
predicted_labels = clf.predict(test_vectors)
precision, recall, f1, support = precision_recall_fscore_support(
    test_labels, predicted_labels)

print('Звичайна модель Логістичної регресії')
print("Precision (рекомендовано/не рекомендовано):", *precision)
print("Recall (рекомендовано/не рекомендовано):", *recall)
print("F1 Score (рекомендовано/не рекомендовано):", *f1)
print("Support (рекомендовано/не рекомендовано):", *support)


# Підберемо оптимальні параметри
params_grid = [
    {'C': [0.1, 1, 10], 'penalty': ['l1'], 'solver': [
        'liblinear', 'saga']},
    {'C': [0.1, 1, 10], 'penalty': ['l2'],
        'solver': ['newton-cg', 'liblinear', 'newton-cholesky', 'saga', 'lbfgs', 'sag']},
]
optimized_model = GridSearchCV(
    LogisticRegression(multi_class='ovr'), params_grid, scoring='recall_macro', n_jobs=-1)
optimized_model.fit(train_vectors, train_labels)
predicted_labels = optimized_model.predict(test_vectors)
precision, recall, f1, support = precision_recall_fscore_support(
    test_labels, predicted_labels)

print('Оптимізована модель Логістичної регресії')
print("Precision (рекомендовано/не рекомендовано):", *precision)
print("Recall (рекомендовано/не рекомендовано):", *recall)
print("F1 Score (рекомендовано/не рекомендовано):", *f1)
print("Support (рекомендовано/не рекомендовано):", *support)

# Застосуємо алгоритм SVC
svm = SVC(kernel='linear')
svm.fit(train_vectors, train_labels)

# Метрики
predicted_labels = svm.predict(test_vectors)
precision, recall, f1_score, support = precision_recall_fscore_support(
    test_labels, predicted_labels, average=None)

print('Звичайна модель SVM')
print("Precision (рекомендовано/не рекомендовано):", precision)
print("Recall (рекомендовано/не рекомендовано):", recall)
print("F1 Score (рекомендовано/не рекомендовано):", f1_score)
print("Support (рекомендовано/не рекомендовано):", support)

# Підберемо оптимальні параметри
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']
              }
grid_search = GridSearchCV(
    SVC(), param_grid, scoring='recall_macro', n_jobs=-1)
grid_search.fit(train_vectors, train_labels)
predicted_labels = grid_search.predict(test_vectors)
precision, recall, f1_score, support = precision_recall_fscore_support(
    test_labels, predicted_labels, average=None)

print('Оптимізована модель SVM')
print("Precision (рекомендовано/не рекомендовано):", precision)
print("Recall (рекомендовано/не рекомендовано):", recall)
print("F1 Score (рекомендовано/не рекомендовано):", f1_score)
print("Support (рекомендовано/не рекомендовано):", support)
# Як ми бачимо, найкращі результати показала нам оптимізована модель Логістичної регресії.
