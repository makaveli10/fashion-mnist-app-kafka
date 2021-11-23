import json
import os
def read_json(fi):
    with open(fi) as f:
        json_dict = json.load(f)
    return json_dict