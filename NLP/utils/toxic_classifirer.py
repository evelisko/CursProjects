import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CheckToxicity():
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.score = 0.9
        self.name_value = []

    def load(self, model_name_or_path: str = None, score: int = 0.9, toxic_colors: dict = {}):
        self.toxic_colors = toxic_colors
        self.name_value =list(self.toxic_colors.keys())
        print(self.name_value)
        self.score = score
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            self.model.cuda()

    def text2toxicity(self, text):
        """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
            result = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()
            print(result)
        if isinstance(text, str):
            result = list(map(lambda y: y[1],
                              list(filter(lambda x: x[0] >= self.score, zip(result[0], self.name_value)))))
            print(result)
            if result:
                index = 1 if result[0] == self.name_value[0] and len(result) > 1 else 0
            else:
                return self.toxic_colors[self.name_value[0]]

            return self.toxic_colors[result[index]]
        else:
            return self.toxic_colors[self.name_value[0]]
