# Распознование капчи 

Данный репозиторий содержит модель CNN+LSTM CTCLoss для распознования текста капчи

Примеры капчи:

![](samples/2cg58.png)

![](samples/d22y5.png)

![](samples/2p2y8.png)

Отчет с описанием этапов и результатами обучения представлен в ноутбуке `notebooks/report.ipynb`

# Оценка

Для оценки использовалась метрика Character Error Rate, значение которой на тестовой выборке составило `0.006` 
