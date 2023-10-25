from dynaconf import Dynaconf


class Config(object):
    def __init__(self, parser: Dynaconf):
        self.parser = parser
        self.token = self.parser.get("token")
