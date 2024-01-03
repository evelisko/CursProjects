import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name_value = ['non-toxic', 'insult', 'obscenity', 'threat', 'dangerous']

toxic_colors = {'non-toxic': '',
                'insult': 'Ох, Дорогой, пожалуйста без грубостей. Или я не буду с тобой разговаривать.',
                'obscenity': 'Ох, Дорогой, кто научил тебя таким словам.',
                'threat': 'Молодой человек! я бы попросила ... Нам лучше прекратить беседу.',
                'dangerous': 'Молодой человек! я бы попросила ... Нам лучше прекратить беседу.'
                }

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
            index = 1 if result[0] == name_value[0] and len(result) > 1 else 0

            return toxic_colors[result[index]]
        else: 
            return toxic_colors['non-toxic']


