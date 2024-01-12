import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import string
import re
from tqdm import tqdm

nltk.download(['stopwords','wordnet'])
stop_words = stopwords.words('russian')

tokenize =  RegexpTokenizer(r'\w+')


def correct_text(text):
    '''
    Приводит текст в нижний регистр, убирает знаки табуляции.
    '''
    if not isinstance(text, str):
        text = "" #str(text)
    
    text = text.lower()
    text = text.strip('\n').strip('\r').strip('\t')
    
    return text


def join_collumns(df, output_collumn, collumns=[]):
    '''
    Объединяет колонки,  записывает результат в output_collumn.
    содержимое разных колонок в результирующей строке разделяется запятой.
    '''
    count = df.shape[0]
    df[output_collumn] = ""
    texts = []
    for ind in tqdm(range(count)):
            text_list = []
            for i in collumns:
                t = df[i][ind]
                if isinstance(t, str) and t !="":
                    t = t.lower()
                    t = t.strip('\n').strip('\r').strip('\t')
                    text_list.append(t)
            if len(text_list) > 0:
                texts.append(' '.join(text_list))
            else:
                texts.append("")
    df[output_collumn] = texts


def clean_text(text):
    '''
    очистка текста
    
    на выходе очищеный текст
    
    '''
    if not isinstance(text, str):
        text = "" #str(text)
    
    text = text.lower()
    text = text.strip('\n').strip('\r').strip('\t')
    text = re.sub("-\s\r\n\|-\s\r\n|\r\n", '', str(text))

    text = re.sub("[0-9]|[-—\.,:;•·&()_%©«»?]|[!@#№$^\*]|[+=]", ' ', text)
    text = re.sub(r"\r\n\t|\n|\\s|\r\t|\\n", ' ', text)
    text = re.sub(r'[\xad]|[\s+]', ' ', text.strip())

    return text

cache = {}

def lemmatization(text):
    '''
    лемматизация
        [1] токенизация предложения
        [2] проверка на стоп-слова
        [3] лемматизация слова
        [4] усановка в качестве разделителей в массиве слов - ","

    на выходе лист отлемматизированых токенов
    '''
    # [1]
    words = tokenize.tokenize(text)
    
    words = [i for i in words if not i in stop_words] # [2]

    lematizer = WordNetLemmatizer() #[3]

    words_lem = ','.join([lematizer.lemmatize(i) for i in words]) #[4]

    return words_lem


def get_tokens(text: str):
    text = clean_text(text)
    text = lemmatization(text)
    tockens = list(set(text.split(',')))
    return tockens

