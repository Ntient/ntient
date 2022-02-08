import pytest
import joblib
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
import torch
import os

def load_sklearn():
    return joblib.load("tests/support/sklearn_model.joblib")

def load_keras():
    return keras.models.load_model("tests/support/keras_model.h5")

def load_pytorch():
    return torch.load("tests/support/pytorch_model.pt")


def test_packager_packages_sklearn(cli_context):
    import ntient
    packager = ntient.Packager(load_sklearn())

    fn = packager.package_sklearn_model()

    os.remove(fn)

def test_packager_packages_keras(cli_context):
    import ntient
    packager = ntient.Packager(load_keras())

    fn = packager.package_keras_model()

    os.remove(fn)

def test_packager_packages_pytorch(cli_context):
    import ntient
    packager = ntient.Packager(load_pytorch())

    fn = packager.package_pytorch_model()

    os.remove(fn)