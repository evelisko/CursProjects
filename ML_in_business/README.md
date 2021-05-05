Итоговый проект (пример) курса "Машинное обучение в бизнесе"

Стек:

**ML:** nltk, re, pandas, dill, sklearn.

**API:** flask

**Данные с kaggle -** https://www.kaggle.com/shivamb/netflix-shows/tasks?taskId=2447

**Задача:** Составить рекоммендации для фильмов. При поиске конкретного фильма.
 
Подробное описание задачи и датасета и файле **Netflix_recomenders.ipynb**!  
data/netflix_titles.csv  - файл с датасетом.  


### Последовательность действий. 

#### 1 . Клонируем репозиторий - 
    
    $ git clone git@github.com:evelisko/CursProjects.git

#### 2. Запустить ноутбук Netflix_recomenders.ipynb. В результате получим будет сформирован бинарный файл - data/tfidf_netflix.dill. 

#### 3. Открыть терминал. Перейти в каталог -  
   
    GeekBrains_CursProjects/ML_in_business/Docker

#### 4. Собрать докер - образ
   
    docker build -t netflix_recomender .


#### 5. Запускаем контейнер

По умолчанию dill файл с моделью сохраняются в каталоге "/GeekBrains_CursProjects/ML_in_business/data" там же должен находиться и файл с дадасетом.

    docker run -d -p 8180:8180 -p 8181:8181 -v <your_catalog>/GeekBrains_CursProjects/ML_in_business/data:/app/app/models netflix_recomender 

#### 6. Проверяем что контейнер запущен -

 docker ps -a

#### 7. Проверяем, что наше api работает

Для этого необходимо выполнить код из ноутбука - Test_recomender.ipynb. 

P.S. Файл конфигураций для сервера расположен -   GeekBrains_CursProjects/ML_in_business/Docker/app. так удобней для работы примера.  
 В последствие его лучше вынести вынести во внешний каталог пределы doker - образа. 
