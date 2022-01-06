import json
from .base import Base
from .model import Model


class API(Base):
    def __init__(self, organization):
        super(API, self).__init__()

        self.organization = organization

        if not organization:
            raise ValueError("Organization is required!")

    def get_model(self, model_name):
        url = f"{self.host}/{self.organization}/ml_model"
        json = self.get_request(url, params={"model_name": model_name})

        model_json = json['ml_models'][0]

        model = Model(
            organization=self.organization,
            name=model_json['name'],
            filename=model_json["filename"],
            model_type=model_json["model_type"],
            existing_model=True,
            input_mapping_json=model_json["input_mapping"],
            output_mapping_json=model_json["output_mapping"]
        )

        model.model_id = model_json["id"]

        return model
