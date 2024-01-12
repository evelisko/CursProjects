import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm

nltk.download(['stopwords', 'wordnet'])
stop_words = stopwords.words('russian')

tokenize = RegexpTokenizer(r'\w+')


def correct_text(text):
    """
    Приводит текст в нижний регистр, убирает знаки табуляции.
    """
    if not isinstance(text, str):
        text = ""  # str(text)

    text = text.lower()
    text = text.strip('\n').strip('\r').strip('\t')

    return text


def join_columns(df, output_column, columns=None):
    """
    Объединяет колонки, записывает результат в output_column.
    Содержимое разных колонок в результирующей строке разделяется запятой.
    """
    if columns is None:
        columns = []
    count = df.shape[0]
    df[output_column] = ""
    texts = []
    for ind in tqdm(range(count)):
        text_list = []
        for i in columns:
            t = df[i][ind]
            if isinstance(t, str) and t != "":
                t = t.lower()
                t = t.strip('\n').strip('\r').strip('\t')
                text_list.append(t)
        if len(text_list) > 0:
            texts.append(' '.join(text_list))
        else:
            texts.append("")
    df[output_column] = texts


def clean_text(text):
    """
    Очистка текста
    на выходе очищенный текст
    """
    if not isinstance(text, str):
        text = ""  # str(text)

    text = text.lower()
    text = text.strip('\n').strip('\r').strip('\t')
    text = re.sub("-\s\r\n\|-\s\r\n|\r\n", '', str(text))

    text = re.sub("[0-9]|[-—\.,:;•·&()_%©«»?]|[!@#№$^\*]|[+=]", ' ', text)
    text = re.sub(r"\r\n\t|\n|\\s|\r\t|\\n", ' ', text)
    text = re.sub(r'[\xad]|[\s+]', ' ', text.strip())

    return text


cache = {}


def lemmatization(text):
    """
    Лемматизация
        [1] токенизация предложения
        [2] проверка на стоп-слова
        [3] лемматизация слова
        [4] усановка в качестве разделителей в массиве слов - ","

    на выходе лист отлемматизированых токенов
    """
    words = tokenize.tokenize(text)  # [1]
    words = [i for i in words if i not in stop_words]  # [2]
    lemmatizer = WordNetLemmatizer()  # [3]
    words_lem = ','.join([lemmatizer.lemmatize(i) for i in words])  # [4]
    return words_lem


def get_tokens(text: str):
    text = clean_text(text)
    text = lemmatization(text)
    tokens = list(set(text.split(',')))
    return tokens
