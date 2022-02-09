import os
from requests.structures import CaseInsensitiveDict
import requests
import json


class Base():
    def __init__(self):
        self.host = os.environ["NTIENT_HOST"]
        self.headers = CaseInsensitiveDict()
        self.headers["Authorization"] = f"Bearer {os.environ['NTIENT_TOKEN']}"

    def get_request(self, url, params=None):
        response = requests.get(url, headers=self.headers, params=params)
        self.check_response(response)
        return response.json()

    def get_file(self, url):
        response = requests.get(url, headers=self.headers)
        self.check_response(response)
        return response.body

    def post_request(self, url, data):
        response = requests.post(
            url, json=data, headers=self.headers)

        self.check_response(response)
        return response.json()

    def post_upload(self, url, files):
        response = requests.post(url, files=files, headers=self.headers)
        self.check_response(response)
        return response.json()

    def patch_request(self, url, data):
        response = requests.patch(
            url, json=data, headers=self.headers)
        self.check_response(response)
        return response.json()

    def check_response(self, response):
        if response.status_code not in [200, 201, 202, 203, 204]:
            raise requests.exceptions.HTTPError("Not Authorized")
