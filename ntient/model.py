import requests
import json
from .base import Base
from .packager import Packager


class Model(Base):

    def __init__(self,
                 model: object = None,
                 organization: str = "",
                 name: str = "",
                 model_type: str = "",
                 input_mapping_json: dict = {},
                 output_mapping_json: dict = {},
                 existing_model: bool = False
                 ):

        super(Model, self).__init__()

        self.organization = organization
        self.name = name
        self.model = model
        self.model_type = model_type
        self.json_input_filename = f"{self.name}_input.json"
        self.json_output_filename = f"{self.name}_output.json"
        self.input_mapping = {}
        self.output_mapping = {}
        self.existing_model = existing_model
        self.deployed = self.existing_model
        self.packager = Packager(model)

        if input_mapping_json:
            self.input_mapping = json.loads(input_mapping_json)

        if output_mapping_json:
            self.output_mapping = json.loads(output_mapping_json)

        allowed_model_types = [
            "sklearn ExtraTreeClassifier",
            "sklearn DecisionTreeClassifier",
            "sklearn OneClassSVM",
            "sklearn MLPClassifier",
            "sklearn RadiusNeighborsClassifier",
            "sklearn KNeighborsClassifier",
            "sklearn ClassifierChain",
            "sklearn MultiOutputClassifier",
            "sklearn OutputCodeClassifier",
            "sklearn OneVsOneClassifier",
            "sklearn OneVsRestClassifier",
            "sklearn SGDClassifier",
            "sklearn RidgeClassifierCV",
            "sklearn RidgeClassifier",
            "sklearn PassiveAggressiveClassifier    ",
            "sklearn GaussianProcessClassifier",
            "sklearn VotingClassifier",
            "sklearn AdaBoostClassifier",
            "sklearn GradientBoostingClassifier",
            "sklearn BaggingClassifier",
            "sklearn ExtraTreesClassifier",
            "sklearn RandomForestClassifier",
            "sklearn BernoulliNB",
            "sklearn CalibratedClassifierCV",
            "sklearn GaussianNB",
            "sklearn LabelPropagation",
            "sklearn LabelSpreading",
            "sklearn LinearDiscriminantAnalysis",
            "sklearn LinearSVC",
            "sklearn LogisticRegression",
            "sklearn LogisticRegressionCV",
            "sklearn MultinomialNB  ",
            "sklearn NearestCentroid",
            "sklearn NuSVC",
            "sklearn Perceptron",
            "sklearn QuadraticDiscriminantAnalysis",
            "sklearn SVC",
            "sklearn DPGMM",
            "sklearn GMM ",
            "sklearn GaussianMixture",
            "sklearn VBGMM",
            "keras",
            "pytorch"
        ]

        if not organization:
            raise ValueError("Organization is required!")

        if not name:
            raise ValueError("Name is required!")

        if not model:
            raise ValueError("Model is required!")

        if not model_type:
            raise ValueError("Model Type is required!")

        if model_type not in allowed_model_types:
            raise ValueError(f"Model Type: {self.model_type} not supported. Supported model types: {','.join(allowed_model_types)}")

    def push(self):
        # create the model
        if not self.existing_model:
            print("CREATING MODEL")
            self.model_id = self.create_model()
        else:
            print("CANNOT OVERWRITE EXISTING MODEL RECORD")

        self.dump_model()

        print("UPLOADING MODEL FILE")
        self.upload_file()

        os.remove(self.filename)

        print("INTROSPECTING MODEL")
        response = self.introspect_model()
        self.input_mapping = response["input_format"]
        self.output_mapping = response["output_format"]

        print("WRITING FORMAT FILES")
        self.write_format_files(self.input_mapping, self.output_mapping)

        print("MODEL PUSHED.")

        print("DESIGN MODEL API")
        self.update_input_mapping()
        self.update_output_mapping()
        self.add_spec()

    def dump_model(self):
        model_general_type = self.model_type.split(" ")[0]
        func = self.packager.function_map[model_general_type]
        self.filename = func()

    def update_mapping(self, mapping):
        keys = list(mapping.keys())
        datatypes = ["int", "float", "str", ""]

        for key in keys:
            print(f"Update Input Field {int(key) + 1}. Leave blank to skip")
            name = input(f"Field {int(key) + 1} Name: ")
            datatype = None
            while datatype not in datatypes:
                datatype = input(f"Field {int(key) + 1} Type (int, float, str): ")

            if name != "":
                mapping[key]["name"] = name

            if datatype != "":
                mapping[key]["type"] = datatype

        return mapping

    def update_input_mapping(self):
        if not self.input_mapping:
            raise Exception("Input Mapping has not been set")

        number_of_fields = len(list(self.input_mapping.keys()))

        print(f"{number_of_fields} fields detected in model input.")

        self.input_mapping = self.update_mapping(self.input_mapping)

        with open(self.json_input_filename, 'w') as f:
            json.dump(self.input_mapping, f)

    def update_output_mapping(self):
        if not self.output_mapping:
            raise Exception("Input Mapping has not been set")

        number_of_fields = len(list(self.output_mapping.keys()))

        print(f"{number_of_fields} fields detected in model output.")

        self.output_mapping = self.update_mapping(self.output_mapping)

        with open(self.json_output_filename, 'w') as f:
            json.dump(self.output_mapping, f)

    def create_model(self):
        url = f"{self.host}/{self.organization}/ml_model"

        data = {
            "name": self.name,
            "model_type": self.model_type,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping
        }

        r = self.post_request(url, data)

        return int(r["id"])

    def upload_file(self):
        url = f"{self.host}/{self.organization}/ml_model/{self.model_id}/upload"

        file = open(self.filename, 'rb')

        data = {'file': file}

        r = self.post_upload(url, data)

    def introspect_model(self):
        url = f"{self.host}/{self.organization}/ml_model/{self.model_id}/introspect"
        r = self.get_request(url)

        return r

    def write_format_files(self, input_format, output_format):

        with open(self.json_input_filename, 'w') as f:
            json.dump(input_format, f)

        with open(self.json_output_filename, 'w') as f:
            json.dump(output_format, f)

    def add_spec(self):
        url = f"{self.host}/{self.organization}/ml_model/{self.model_id}"
        input_format_file = open(self.json_input_filename)
        output_format_file = open(self.json_output_filename)

        self.input_mapping = json.load(input_format_file)
        self.output_mapping = json.load(output_format_file)

        data = {
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping
        }

        r = self.patch_request(url, data)

    def deploy(self, deployment_name: str, environment: str, v_cores: int = 1, instances: int = 1):
        url = f"{self.host}/{self.organization}/deployment"

        data = {
            "name": deployment_name,
            "environment": environment,
            "instances": instances,
            "v_cores": v_cores,
            "ml_model_id": self.model_id
        }

        r = self.post_request(url, data)

        self.deployed = True

        # if r.status == 200:
        #     print("Deployment Initialized")
