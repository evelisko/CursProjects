import dill
import flask
import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel


class Recomender():

    def __init__(self, model_path, df_path, logger):
        with open(model_path, 'rb') as f:
            self.model = dill.load(f)
        with open(df_path, 'rb') as f:
            self.data_frame = pd.read_csv(f)
        self.sig = sigmoid_kernel(self.model, self.model)
        self.indices = pd.Series(self.data_frame.index, index=self.data_frame['title']).drop_duplicates()
        self.logger = logger

    def recommend(self, title, n):
        try:
            idx = self.indices[title]
            self.logger.write(f"Data: title={title}")
            sim_scores = list(enumerate(self.sig[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n + 1]
            movies_indices = [i[0] for i in sim_scores]
            titles = []
            for title in self.data_frame['title'].iloc[movies_indices]:
                titles.append(title)
            return titles
        except Exception as e:
            self.logger.write(f'Exception: {str(e)}')
            return []

    def search_moves(self, text_title=''):
        try:
            titles = []
            text_title = text_title.lower()
            for title in self.data_frame['title']:
                title_lower = title.lower()
                if (title_lower.find(text_title) > -1):
                    titles.append(title)
            return titles
        except Exception as e:
            self.logger.write(f'Exception: {str(e)}')
            return []

    def search_films(self, request_json):
        titles = []
        if request_json["title"]:
            title = request_json["title"]

            self.logger.write(f'Search film witch title ={title}')
            titles = self.search_moves(title)
        return titles

    def film_info(self, request_json):
        description_film = ""
        if request_json["title"]:
            title = request_json["title"]
            try:
                description_film = self.data_frame.loc[(self.data_frame["title"] == title), "description"].values[0]
            except AttributeError as e:
                self.logger.write(f'Exception: {str(e)}')
        return description_film

    def get_recomendations(self, request_json):
        recomender_titles = []
        count = 5
        self.logger.write(f'metod get_similarity moves')
        if request_json["title"]:
            title = request_json["title"]

            if request_json["count"]:
                count = request_json["count"]

            self.logger.write(f'get_similarity moves for title = {title} count = {count}')
            try:
                recomender_titles = self.recommend(title, count)

            except AttributeError as e:
                self.logger.write(f'Exception: {str(e)}')

        return recomender_titles
