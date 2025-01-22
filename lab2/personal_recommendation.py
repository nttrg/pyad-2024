import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

# 1. Находим нужного пользователя
user_with_most_zeros = (
    df_ratings[df_ratings["Book-Rating"] == 0]
    .groupby("User-ID")["Book-Rating"]
    .count()
    .idxmax()
)

# 2. Делаем предсказание SVD для книг, которым он "поставил" 0
user_zero_ratings =df_ratings[
    (df_ratings["User-ID"] == user_with_most_zeros) & (df_ratings["Book-Rating"] == 0)
]

with open("svd_model.pkl", "rb") as svd_file:
    svd_model = pickle.load(svd_file)

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)

books_to_predict = user_zero_ratings["ISBN"].unique()
predicted_ratings_svd = [
    (isbn, svd_model.predict(user_with_most_zeros, isbn).est)
    for isbn in books_to_predict
]

#  3. Берем те книги, для которых предсказали рейтинг не ниже 8.
high_rated_books = [isbn for isbn, rating in predicted_ratings_svd if rating >= 8]

# 4. Делаем предсказание LinReg для этих же книг
with open("linreg_model.pkl", "rb") as linreg_file:
    linreg_model = pickle.load(linreg_file)

high_rated_books_data = df_books[df_books["ISBN"].isin(high_rated_books)]
X_high_rated = high_rated_books_data[
    ["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]
]
predicted_ratings_linreg = linreg_model.predict(X_high_rated)

# 5. Сортируем полученный на шаге 3 список по убыванию рейтинга линейной модели
recommendations = list(
    zip(high_rated_books_data["Book-Title"], predicted_ratings_linreg)
)
recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

# 6. В конце файла комментарием записываем полученную рекомендацию
print("Рекомендации для пользователя:")
for title, rating in recommendations:
    print(f"{title}: {rating:.2f}")
