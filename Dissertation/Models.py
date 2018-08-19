class Models(object):
    def __init__(self):
        self.models = {}

    def get_models(self):
        return self.models

    def add_model(self, model, model_name):
        self.models[model_name] = model

    def get_model(self, model_name):
        return self.models[model_name]





