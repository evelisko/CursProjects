import pickle
import nmslib
import numpy as np
import pandas as pd
from gensim.models.fasttext import FastText
from utils.dataset_prepare import get_tokens


class SmartSearch:
    def __init__(self):
        self.recipes_key_words = None
        self.recommend_count = None
        self.max_distance = None
        self.fastchat_model = None
        self.dataset = None
        self.vector_db = None
        self.index = None

    def set_config(self,
                   dataset_path: str,
                   vectorizer_model_path: str,
                   vector_db_path: str,
                   max_distance: float = 0.2,
                   recommend_count: int = 2,
                   recipes_key_words=None
                   ):
        if recipes_key_words is None:
            recipes_key_words = []
        self.max_distance = max_distance
        self.recommend_count = recommend_count
        self.fastchat_model = FastText.load(vectorizer_model_path)
        self.dataset = pd.read_json(dataset_path, orient='table', encoding='utf-8')
        self.recipes_key_words = recipes_key_words

        with open(vector_db_path, "rb") as file:
            vectors = np.vstack(pickle.load(file))

        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(vectors)
        self.index.createIndex({'post': 2}, print_progress=True)

    def find_recipes(self, text: str, check: bool = True):
        question = text
        instructions = []
        if check:
            question = self.check_recipes(text)
        if question:
            text_tokens = get_tokens(question)
            query = [self.fastchat_model.wv[vec] for vec in text_tokens]
            query = np.mean(query, axis=0)

            ids, distances = self.index.knnQuery(query, k=self.recommend_count)
            for i, distance in zip(ids, distances):
                if distance <= self.max_distance:
                    print(f"distance: {distance:.2f},\t {self.dataset['name'].values[i]}")
                    instructions.append(self.dataset['instructions'].values[i])

        recipies = []
        if len(instructions) > 1:
            for i, dish in enumerate(instructions):
                recipies.append(f'#Рецепт №{i + 1}: {dish}')
            recipies = '\r\n'.join(recipies)
        elif instructions:
            recipies = instructions[0]
        return recipies

    def check_recipes(self, text: str):
        '''Проверяем есть ли в запросе просьба узнать рецепт какого-либо блюда.'''
        result = ''
        for key_word in self.recipes_key_words:
            if key_word in text.lower():
                print(key_word)
                result = text.lower().split(key_word)[-1].strip()
                break
        print(f'check_recipes {result}')
        return result
