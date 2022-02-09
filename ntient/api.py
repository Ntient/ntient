import json
from .base import Base
from .model import Model
from .user import User

class AuthorizationException(Exception):
    pass


class API(Base):
    def __init__(self, organization):
        super(API, self).__init__()

        self.organization = organization

        if not organization:
            raise ValueError("Organization is required!")

    def get_model(self, model_id):
        url = f"{self.host}/{self.organization}/ml_model/{model_id}"
        model_json = self.get_request(url)

        model = Model(
            organization=self.organization,
            name=model_json['name'],
            filename=model_json["filename"],
            model_type=model_json["model_type"],
            existing_model=True,
            input_mapping_json=model_json["input_mapping"],
            output_mapping_json=model_json["output_mapping"],
            s3_path=model_json["s3_path"]
        )

        model.model_id = model_json["id"]

        return model

    def authorize_client_for_deployment(self, deployment_id, token=None, client_id=None, client_secret=None):
        if token is None and client_id is None and client_secret is None:
            raise AuthorizationException

        if token:
            url = f"{self.host}/{self.organization}/deployment/{deployment_id}"
            self.headers["Authorization"] = f"Bearer {token}"
            try:
                self.get_request(url)
            except:
                raise AuthorizationException

            return True
        
        if client_id:
            url = f"{self.host}/{self.organization}/deployment/{deployment_id}/aplication/authorize"
            self.headers["CLIENT_ID"] = client_id
            self.headers["CLIENT_SECRET"] = client_secret
            self.headers["Authorizzation"] = ""
            try:
                resp = self.get_request(url)
                assert resp["message"] == "Access Granted"
            except:
                raise AuthorizationException

            return True

        


