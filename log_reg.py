import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
import time

class LinReg:
    def __init__(self, learning_rate=0.1, regularization=None, alpha=0.01, l1_ratio=0.5):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.history = []

    def sigmoida(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_epochs = 1000
        X = np.array(X)
        y = np.array(y)

        self.coef_ = np.random.normal(size=X.shape[1])  # Инициализируем веса
        self.intercept_ = np.random.normal()  # Инициализируем свободный член w0

        for i in range(n_epochs):
            z = self.intercept_ + X @ self.coef_
            p = self.sigmoida(z)
            error = (p - y)

            w0_grad = error
            w_grad = X * error.reshape(-1, 1)

            if self.regularization == 'l1':
                self.coef_ -= self.learning_rate * (w_grad.mean(axis=0) + self.alpha * np.sign(self.coef_))
            elif self.regularization == 'l2':
                self.coef_ -= self.learning_rate * (w_grad.mean(axis=0) + self.alpha * self.coef_)
            elif self.regularization == 'elasticnet':
                l1_term = self.alpha * self.l1_ratio * np.sign(self.coef_)
                l2_term = self.alpha * (1 - self.l1_ratio) * self.coef_
                self.coef_ -= self.learning_rate * (w_grad.mean(axis=0) + l1_term + l2_term)
            else:
                self.coef_ -= self.learning_rate * w_grad.mean(axis=0)

            self.intercept_ -= self.learning_rate * w0_grad.mean()
            self.history.append((self.coef_.copy(), self.intercept_))

    def predict(self, X):
        X = np.array(X)
        probabilities = self.sigmoida(self.intercept_ + X @ self.coef_)
        return np.column_stack([1 - probabilities, probabilities])

    def score(self, X, y):
        predictions = self.predict(X)
        y_pred = (predictions[:, 1] > 0.5).astype(int)  # Бинарные предсказания на основе вероятностей
        accuracy = accuracy_score(y, y_pred)
        return accuracy

    def plot_decision_boundary(self, train, feature1, feature2, update_interval=10):
        plot_placeholder = st.empty()

        for epoch, (coef, intercept) in enumerate(self.history):
            if epoch % update_interval == 0:  # Обновление каждые `update_interval` эпох
                plt.figure(figsize=(10, 10))
                colors = {0: 'blue', 1: 'red'}
                plt.scatter(train[feature1], train[feature2], c=train['Personal.Loan'].apply(lambda x: colors[x]))

                x = np.linspace(train[feature1].min(), train[feature1].max(), 1000)
                y = -(intercept + coef[0] * x) / coef[1]

                plt.plot(x, y, color='grey', label='prediction')  # Прямая предсказания нашего алгоритма
                plt.title(f'{epoch}-ая эпоха обучения')
                plt.legend()
                plt.xlim(train[feature1].min() - 1, train[feature1].max() + 1)
                plt.ylim(train[feature2].min() - 1, train[feature2].max() + 1)
                # plt.pause(0.01)
                # clear_output(wait=True)
                plot_placeholder.pyplot(plt.gcf())  # Обновляем график в контейнере
                plt.close()
                time.sleep(0.01)

# Streamlit app
st.title("Логистическая регрессия")
st.subheader("Загрузите датафрейм и выберите необходимые признаки для нормировки и регресии")

# Загрузка файла CSV
uploaded_file = st.sidebar.file_uploader("Загрузить CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Датасет до нормировки:")
    st.write(df.head())
    

    # Выбор фичи для нормировки
    feature1n = st.selectbox("Выберите первую фичу для нормировки", df.columns)
    feature2n = st.selectbox("Выберите вторую фичу для нормировки", df.columns)

    # Нормализация данных
    scaler = StandardScaler()
    df[[feature1n, feature2n]] = scaler.fit_transform(df[[feature1n, feature2n]])

    st.write("Нормированные данные:")
    st.write(df.head())

    # Выбор метода регуляризации
    regularization = st.selectbox("Выберите метод регуляризации", ["None", "l1", "l2", "elasticnet"])

    # Выбор фич для визуализации
    feature1 = st.selectbox("Выберите первую фичу", df.columns)
    feature2 = st.selectbox("Выберите вторую фичу", df.columns)
    target = st.selectbox("Выберите таргет", df.columns)

    # Инициализация и обучение модели
    X = df[[feature1, feature2]].values
    y = df[target].values

    model = LinReg(regularization=regularization if regularization != "None" else None)
    model.fit(X, y)

    # Вывод результатов регрессии в виде словаря
    results = {feature: weight for feature, weight in zip([feature1, feature2], model.coef_)}
    st.write("Результаты регрессии:", results)
    st.write('Свободный член:', model.intercept_)
    if np.all(np.isin(y, [0, 1])):
        st.write("Точность Accuracy", model.score(X, y=df[target]))


    # Кнопка для построения графика
    if st.button("Построить график"):
        model.plot_decision_boundary(df, feature1, feature2, update_interval=20)
