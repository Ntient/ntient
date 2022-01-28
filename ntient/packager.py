import joblib
import uuid

class Packager():
    def __init__(self, model):
        self.model = model

        self.function_map = {
            "sklearn": self.package_sklearn_model,
            "keras": self.package_keras_model,
            "pytorch": self.package_pytorch_model
        }

    def package_sklearn_model(self):
        filename = str(uuid.uuid1()) + ".joblib"
        joblib.dump(self.model, filename)

        return filename

    def package_keras_model(self):
        filename = str(uuid.uuid1()) + ".h5"
        self.model.save(filename)

        return filename

    def package_pytorch_model(self):
        import torch
        filename = str(uuid.uuid1()) + ".pytorch"
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(filename)

        return filename


