import pytest
import requests
import json

def model_response():
    return {
        "id": 1,
        "user_id": 1,
        "s3_path": "dummy_path",
        "name": "test_nam e",
        "filename": "test_filename",
        "model_type": "sklearn DecisionTreeClassifier",
        "deployments": [],
        "monthly_usage": 0.0,
        "total_usage": 0.0,
        "input_mapping": {},
        "output_mapping": {}
    }

def setup_response():
    resp = requests.Response()
    resp._content = bytes(json.dumps(model_response()), 'utf-8')
    resp.status_code = 200

    return resp

def test_api_initialization(cli_context):
    import ntient
    ntient.API("organization")

def test_get_model_makes_an_api_call(cli_context, mocker):
    import ntient
    client = ntient.API("organization")

    requests_mock = mocker.patch.object(
        requests, "get", return_value=setup_response()
    )

    client.get_model(1)

    requests_mock.assert_called_with(
        "ntient.ai/api/organization/ml_model/1",
        headers={"Authorization": f"Bearer test_token"}, params=None)

def test_authorize_errors_without_any_creds(cli_context):
    import ntient
    client = ntient.API("organization")

    with pytest.raises(ntient.api.AuthorizationException) as exception:
        client.authorize_client_for_deployment(1)

def test_authorize_works_with_token(cli_context, mocker):
    import ntient
    client = ntient.API("organization")

    resp = requests.Response()
    resp.status_code = 200

    return resp

    requests_mock = mocker.patch.object(
        requests, "get", return_value=resp
    )

    client.authorize_client_for_deployment(1, token="test_token")

    requests_mock.assert_called()

def test_authorize_errors_if_token_is_invalid(cli_context, mocker):
    import ntient
    client = ntient.API("organization")

    resp = requests.Response()
    resp.status_code = 401

    return resp

    requests_mock = mocker.patch.object(
        requests, "get", return_value=resp
    )

    with pytest.raises(ntient.api.AuthorizationException) as exception:
        client.authorize_client_for_deployment(1, token="test_token")

def test_authorize_works_with_client_id_client_secret(cli_context, mocker):
    import ntient
    client = ntient.API("organization")

    resp = requests.Response()
    resp.status_code = 200

    return resp

    requests_mock = mocker.patch.object(
        requests, "get", return_value=resp
    )

    client.authorize_client_for_deployment(1, client_id="client_id", client_secret="client_secret")

    requests_mock.assert_called()

def test_authorize_errors_if_client_id_client_secret_not_authorized(cli_context, mocker):
    import ntient
    client = ntient.API("organization")

    resp = requests.Response()
    resp.status_code = 401

    return resp

    requests_mock = mocker.patch.object(
        requests, "get", return_value=resp
    )

    with pytest.raises(ntient.api.AuthorizationException) as exception:
        client.authorize_client_for_deployment(1, client_id="client_id", client_secret="client_secret")