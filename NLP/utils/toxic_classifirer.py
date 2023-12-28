import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name_value = ['non-toxic', 'insult', 'obscenity', 'threat', 'dangerous']
#            [нетоксичен, оскорбление, непристойность, угроза, опасность]
# В зависимости от типа оскорбления. бот использует разные ключевые фразы.
# Создадим файл с заготовленными фразами.
#----------------------------

class CheckToxicity(): # Посмотреть сколько ресурсов потребляет модель.
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.score = 0.9

    def load(self, model_name_or_path: str = None, score: int = 0.9):
        self.score = score
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            self.model.cuda()

    def text2toxicity(self, text):
        """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
            result = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()
            print(result)
        if isinstance(text, str):
            result = list(map(lambda y: y[1],
                        list(filter(lambda x: x[0] >= self.score, zip(result[0], name_value)))))
            print(result) 
            # если есть текст кроме нетоксичности, значит он токсичный.
            # выбираем максимально значение из токсичных. И вычисляем для него 
        return result


