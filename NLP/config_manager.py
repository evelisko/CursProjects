from dynaconf import Dynaconf


class Config(object):
    def __init__(self, parser: Dynaconf):
        self.parser = parser
        self.token = self.parser.get("token")
        self.llm_model = self.parser.get("llm_model")
        self.classifire_model = self.parser.get("classifire_model")
        self.system_prompt = self.parser.get("system_prompt")
        self.toxicity_score = self.parser.get("toxicity_score")
        self.is_lora = self.parser.get("is_lora")
        self.is_4bit = self.parser.get("is_4bit")
        self.toxic_colors = self.parser.get("toxic_colors")

        
