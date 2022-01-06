# type: ignore
import pytest
from pytest_mock import mocker
import json
import requests
import os


def setup_response(data):
    resp = requests.Response()
    resp._content = bytes(json.dumps(data), 'utf-8')
    resp.status_code = 200

    return resp


def test_model_does_not_init_without_organization_name(cli_context):
    import ntient
    with pytest.raises(ValueError) as exception:
        ntient.Model()

    assert str(exception.value) == "Organization is required!"


def test_model_does_not_init_without_name(cli_context):
    import ntient
    with pytest.raises(ValueError) as exception:
        ntient.Model(
            organization="Test Org"
        )

    assert str(exception.value) == "Name is required!"


def test_model_does_not_init_without_filename(cli_context):
    import ntient
    with pytest.raises(ValueError) as exception:
        ntient.Model(
            organization="Test Org",
            name="Test Name"
        )

    assert str(exception.value) == "Filename is required!"


def test_model_does_not_init_without_model_type(cli_context):
    import ntient
    with pytest.raises(ValueError) as exception:
        ntient.Model(
            organization="Test Org",
            name="Test Name",
            filename="testfile.txt"
        )

    assert str(exception.value) == "Model Type is required!"


def test_model_does_not_init_if_model_type_not_allowed(cli_context):
    import ntient
    with pytest.raises(ValueError) as exception:
        ntient.Model(
            organization="Test Org",
            name="Test Name",
            filename="testfile.txt",
            model_type="NOT ALLOWED"
        )

    assert "Model Type: NOT ALLOWED not supported." in str(exception.value)


def test_model_sends_request_to_server_for_creation(cli_context, mocker):
    import ntient
    resp = setup_response({"id": 1})
    requests_mock = mocker.patch.object(
        requests, "post", return_value=resp)

    model = ntient.Model(
        organization="test_org",
        name="Test Name",
        filename="keras_model.zip",
        model_type="keras"
    )

    input_json = {'name': model.name, 'model_type': model.model_type,
                  'input_mapping': {}, 'output_mapping': {}}

    model.create_model()

    requests_mock.assert_called_with(
        "ntient.ai/api/test_org/ml_model",
        json=input_json,
        headers={"Authorization": f"Bearer test_token"})


def test_model_sends_request_to_server_for_uploading_file(cli_context, mocker):
    import ntient
    resp = setup_response({})
    requests_mock = mocker.patch.object(
        requests, "post", return_value=resp
    )

    model = ntient.Model(
        organization="test_org",
        name="Test Name",
        filename="keras_model.zip",
        model_type="keras"
    )

    model.model_id = 1

    input_file = open("keras_model.zip", "wb")
    data = {"file": input_file}

    model.upload_file()

    # Eventually figure out how to get this to path with arguments included
    requests_mock.assert_called()


def test_introspect_response_is_handled_correctly(cli_context, mocker):
    import ntient
    input_json = json.load(open("tests/test_input_format.json"))
    output_json = json.load(open("tests/test_output_format.json"))
    data = {
        "input_format": input_json,
        "output_format": output_json
    }

    resp = setup_response(data)

    requests_mock = mocker.patch.object(
        requests, "get", return_value=resp
    )

    model = ntient.Model(
        organization="test_org",
        name="Test Name",
        filename="keras_model.zip",
        model_type="keras"
    )

    model.model_id = 1

    response = model.introspect_model()
    url = "ntient.ai/api/test_org/ml_model/1/introspect"

    requests_mock.assert_called_with(
        url,
        headers={"Authorization": f"Bearer test_token"},
        params=None
    )

    assert response == json.load(open("tests/test_generated_format.json"))


def test_introspect_writes_files(cli_context, mocker):
    import ntient
    input_json = json.load(open("tests/test_input_format.json"))
    output_json = json.load(open("tests/test_output_format.json"))

    data = {
        "input_format": input_json,
        "output_format": output_json
    }

    resp = setup_response(data)

    requests_mock = mocker.patch.object(
        requests, "get", return_value=resp
    )

    model = ntient.Model(
        organization="test_org",
        name="Test Name",
        filename="keras_model.zip",
        model_type="keras"
    )

    model.model_id = 1

    response = model.introspect_model()
    model.write_format_files(
        response["input_format"], response["output_format"])

    assert os.path.exists(f"{model.name}_input.json")
    assert os.path.exists(f"{model.name}_output.json")

    generated_format = json.load(open("tests/test_generated_format.json"))

    assert json.load(open(f"{model.name}_input.json")) == generated_format["input_format"]
    assert json.load(open(f"{model.name}_output.json")) == generated_format["output_format"]

    os.remove(f"{model.name}_input.json")
    os.remove(f"{model.name}_output.json")


def test_model_deploy_calls_api_with_proper_format(cli_context, mocker):
    import ntient
    resp = setup_response({})
    requests_mock = mocker.patch.object(
        requests, "post", return_value=resp
    )

    model = ntient.Model(
        organization="test_org",
        name="Test Name",
        filename="keras_model.zip",
        model_type="keras"
    )

    model.model_id = 1

    input_data = {
        "name": "test_deployment",
        "environment": "sandbox",
        "instances": 1,
        "v_cores": 1,
        "ml_model_id": 1
    }

    model.deploy("test_deployment", "sandbox", v_cores=1, instances=1)
    url = "ntient.ai/api/test_org/deployment"
    requests_mock.assert_called_with(
        url,
        json=input_data,
        headers={"Authorization": f"Bearer test_token"}
    )
