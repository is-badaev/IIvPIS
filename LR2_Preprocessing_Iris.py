# Импорт библиотек
import pandas as pd                    # Работа с таблицами данных
import numpy as np                     # Математические операции
import matplotlib.pyplot as plt        # Визуализация графиков

# Загрузка набора данных (предварительно должен быть скачан и сохранён в одной папке с .py файлом)
# Замените путь на путь к вашему файлу, если он отличается
data = pd.read_csv("iris.csv")  # Предполагается, что файл 'iris.csv' находится в той же директории

# Отображение первых строк набора данных
print("Первые 5 строк данных:")
print(data.head())

# Общая информация о наборе данных
print("\nОбщая информация:")
print(data.info())

# Статистические характеристики признаков
print("\nСтатистические характеристики:")
print(data.describe())

# Проверка на наличие пропущенных значений
print("\nПропущенные значения:")
print(data.isnull().sum())

# Оценка основных свойств: средние, минимальные, максимальные значения
print("\nСредние значения признаков:")
print(data.mean(numeric_only=True))

print("\nМинимальные значения признаков:")
print(data.min(numeric_only=True))

print("\nМаксимальные значения признаков:")
print(data.max(numeric_only=True))

# Визуализация: гистограммы распределения всех числовых признаков
data.hist(figsize=(10, 8))
plt.suptitle("Гистограммы признаков набора данных")
plt.show()

# Визуализация: диаграммы рассеяния для всех пар признаков
pd.plotting.scatter_matrix(data, figsize=(12, 10), diagonal='hist')
plt.suptitle("Диаграммы рассеяния по парам признаков")
plt.show()

# Исключаем Id вручную и оставляем только признаки
columns_to_plot = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
data[columns_to_plot].boxplot(figsize=(10, 7))
plt.title("Boxplot диаграммы признаков (без Id)")
plt.show()

# Группировка данных по целевому признаку (например, вид цветка)
if 'species' in data.columns:
    print("\nСредние значения по классам (вид цветка):")
    print(data.groupby("species").mean(numeric_only=True))
else:
    print("\nЦелевой признак 'species' отсутствует в данных.")
