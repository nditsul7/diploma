#Інтелектуальна система виявлення аномалій та шахрайства у великих масивах транзакційних даних

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![Status](https://img.shields.io/badge/status-diploma%20project-green)

Дипломний проєкт: інтелектуальна програмна система, яка автоматично виявляє аномальні та шахрайські операції у великих масивах транзакційних даних за допомогою методів машинного навчання.

## Опис

Система обробляє великі обсяги банківських транзакцій, формує інформативні ознаки та визначає ймовірність шахрайства. Вона поєднує навчання з учителем (класифікація відомих випадків шахрайства) і виявлення аномалій (пошук нетипових операцій). Результати доступні через REST API та інтерактивну панель керування.

## Можливості

- Обробка великих масивів транзакційних даних (датасет Credit Card Fraud Detection та синтетичні дані)
- Формування ознак (feature engineering)
- Виявлення шахрайських транзакцій (класифікація)
- Виявлення аномальних операцій
- Робота з незбалансованими класами
- REST API для перевірки транзакцій у реальному часі
- Інтерактивна панель керування з історією перевірок
- Запуск у Docker однією командою

## 🧰 Технології

| Напрям             | Інструменти              |
|--------------------|--------------------------|
| Мова               | Python                   |
| Машинне навчання   | scikit-learn, pandas     |
| Серверна частина   | FastAPI                  |
| Інтерфейс          | Streamlit                |
| Сховище            | SQLite                   |
| Розгортання        | Docker, docker-compose   |
| Тестування         | pytest                   |

> Залиште лише те, що справді використовується.

## 📂 Структура проєкту

```
fraud_detection_system/
├── app/
│   ├── main.py                 # API (серверна частина)
│   └── dashboard.py            # панель керування
├── src/
│   ├── generate_data.py        # генерація даних
│   ├── feature_engineering.py  # формування ознак
│   ├── pipeline.py             # конвеєр обробки даних
│   └── train.py                # навчання моделі
├── data/                       # дані та налаштування користувачів
├── models/                     # збережені моделі
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

##Швидкий старт

```bash
git clone https://github.com/nditsul7/diploma.git
cd diploma/fraud_detection_system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

##Дані

Через великий розмір файл `creditcard.csv` не включений до репозиторію.

1. Завантажте датасет [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) з Kaggle.
2. Покладіть файл `creditcard.csv` у папку `data/datasets/admin/`.

##Запуск

**Навчання моделі:**

```bash
python src/train.py
```

**API** (документація: http://localhost:8000/docs):

```bash
uvicorn app.main:app --reload
```

**Панель керування:**

```bash
streamlit run app/dashboard.py
```

**Через Docker:**

```bash
docker-compose up --build
```


## Автор

**Діцул Микола**, ДТЕУ.
