# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка отображения графиков
plt.style.use("seaborn-v0_8")
sns.set(font_scale=1.1)

# Загрузка набора данных
df = pd.read_csv("Iris.csv")

# Первичный анализ данных
print("Структура данных:")
print(df.info())
print("\nСтатистика по признакам:")
print(df.describe())
print("\nРаспределение по видам ирисов:")
print(df["Species"].value_counts())

# Распределение количественных признаков
df.hist(column=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], figsize=(10, 8))
plt.suptitle("Распределение количественных признаков")
plt.show()

# Визуализация распределений с использованием гистограмм
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="SepalLengthCm", hue="Species", kde=True, bins=20)
plt.title("Распределение длины чашелистика (SepalLengthCm)")
plt.xlabel("Длина чашелистика (см)")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="PetalWidthCm", hue="Species", kde=True, bins=20)
plt.title("Распределение ширины лепестка (PetalWidthCm)")
plt.xlabel("Ширина лепестка (см)")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

# Зависимость длины лепестка от длины чашелистика
plt.figure(figsize=(6, 4))
plt.plot(df["SepalLengthCm"], df["PetalLengthCm"], 'o')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Зависимость длины лепестка от длины чашелистика")
plt.grid()
plt.show()

# Диаграммы рассеяния (scatterplot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="SepalLengthCm", y="PetalLengthCm", hue="Species", style="Species", s=100)
plt.title("Связь длины чашелистика и лепестка по видам")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")
plt.grid(True)
plt.show()

# Countplot — распределение по видам
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Species")
plt.title("Количество экземпляров каждого вида")
plt.xlabel("Вид ириса")
plt.ylabel("Количество")
plt.grid(True)
plt.show()

# Попарные зависимости между признаками (pairplot)
sns.pairplot(df.drop(columns="Id"), hue="Species", diag_kind="hist")
plt.suptitle("Попарные зависимости между признаками", y=1.02)
plt.show()

# Корреляционная матрица (только числовые признаки)
correlation = df.drop(columns=["Id", "Species"]).corr()
print("\nМатрица корреляции признаков:")
print(correlation)

# Тепловая карта (heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Матрица корреляции количественных признаков")
plt.show()
