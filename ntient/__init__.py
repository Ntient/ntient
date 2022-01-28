from .model import Model
from .api import API
from .packager import Packager
import os
import getpass

if not os.environ.get("NTIENT_TOKEN"):
    print("Environment variable NTIENT_TOKEN not found.")

    token = getpass.getpass(
        prompt="Please input your NTIENT token. It can be found on the application home page. ")

    os.environ["NTIENT_TOKEN"] = token

if not os.environ.get("NTIENT_HOST"):
    print("Environment variable NTIENT_HOST not found.")
    host = None
    while True:
        host = input(
            "Please input your NTIENT host from application home page. Leave blank to use default https://api.ntient.ai/api. ")

        print(host)
        if ("http://" in host or "https://" in host) and "/api" in host:
            break

        if host == "" or host is None:
            break

    if host == "":
        host = "https://api.ntient.ai"

    os.environ["NTIENT_HOST"] = host
