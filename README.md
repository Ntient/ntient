# <img src="https://ntient.ai/wp-content/uploads/2022/01/NTIENT-logo-Horizontal-orange-gradient-highres.png" width=250  >

NTIENT allows you to deploy and integrate AI models in minutes. This package interfaces with the API to manage models, deployments, and APIs.

**Version**: 0.1.6

**Test Status**: <img src="https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoicUJERHZCblBMcm9rMWdlUXRSNVhBdmhNQXMrVFZZMDVjWDBGMDlWTDRvQm04bmFlNVVpb2F1OHB2ekdDVXFWemtEaU9wd0d0VGNZZVd1WW40Vy85NzVJPSIsIml2UGFyYW1ldGVyU3BlYyI6IjhUQVNuWmcxQUdDMUdNR0IiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master" >

Install with pip: `pip install ntient`

# Support Model Types
- Scikit-Learn
  - sklearn ExtraTreeClassifier
  - sklearn DecisionTreeClassifier
  - sklearn OneClassSVM
  - sklearn MLPClassifier
  - sklearn RadiusNeighborsClassifier
  - sklearn KNeighborsClassifier
  - sklearn ClassifierChain
  - sklearn MultiOutputClassifier
  - sklearn OutputCodeClassifier
  - sklearn OneVsOneClassifier
  - sklearn OneVsRestClassifier
  - sklearn SGDClassifier
  - sklearn RidgeClassifierCV
  - sklearn RidgeClassifier
  - sklearn PassiveAggressiveClassifier
  - sklearn GaussianProcessClassifier
  - sklearn VotingClassifier
  - sklearn AdaBoostClassifier
  - sklearn GradientBoostingClassifier
  - sklearn BaggingClassifier
  - sklearn ExtraTreesClassifier
  - sklearn RandomForestClassifier
  - sklearn BernoulliNB
  - sklearn CalibratedClassifierCV
  - sklearn GaussianNB
  - sklearn LabelPropagation
  - sklearn LabelSpreading
  - sklearn LinearDiscriminantAnalysis
  - sklearn LinearSVC
  - sklearn LogisticRegression
  - sklearn LogisticRegressionCV
  - sklearn MultinomialNB
  - sklearn NearestCentroid
  - sklearn NuSVC
  - sklearn Perceptron
  - sklearn QuadraticDiscriminantAnalysis
  - sklearn SVC
  - sklearn DPGMM
  - sklearn GMM
  - sklearn GaussianMixture
  - sklearn VBGMM
- Deep Learning
  - keras
  - pytorch
- Computer Vision
  - yoloV5

# Usage
To use the package, you'll need to set a couple of environment variables.

- NTIENT_HOST
- NTIENT_TOKEN

Both of these can be gathered from the [app](https://app.ntient.ai). Using the package in a Jupyter Notebook will give you the chance to enter both of these, but if you plan to use the package headless, you'll need to set these in your environment beforehand.

# Examples

## Create a Model
Creating models is a foundational piece of the platform. This is the first step in deploying a model. There are two ways you can push a model: Jupyter, and `*.py` file. Both require you to have the trained model object available. You'll need your organization_id, which can be retrieved from the [app](https://app.ntient.ai) as well.

### Jupyter:
Using the package in Jupyter is the simplest setup. The package will prompt you to create the API specs as a part of the `.push()` command.
```
import ntient
...
# train model
...
model = ntient.Model(
    model={trained_model},
    organization={organization_id},
    name={model_name},
    model_type="sklearn DecisionTreeClassifier"
)
model.push()
```

### Script
Using ntient in a script requires you to create the input and output mappings as dicts beforehand. Currently introspection is not supported in the package, so you have to know the input and output formats of your model.
```
import ntient
...
# train model
# Define input/output dicts
...
model = ntient.Model(
    model={trained_model},
    organization={organization_id},
    name={model_name},
    model_type="sklearn DecisionTreeClassifier",
    input_mapping_json=input_mapping_dict,
    output_mapping_json=output_mapping_dict
)
model.push()
```

## Deploy a Model

```
model.deploy(environment={environment_name})
```

# Methods

### `ntient.Model`

- orgainzation: str
- name: str
- model: dumped_model (currently supported: sklearn, pytorch, keras)
- input_mapping_json: dict
- output_mapping_json: dict
- existing_model: bool (use if you're updating an existing model)

### `.deploy`

- deployment_name: str
- environment: str
- v_cores: int (default = 1)
- instances: int (default = 1)


