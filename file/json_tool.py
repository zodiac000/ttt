import json

def load_json(json_file):
    json_data = None
    with open(json_file, 'r') as fp:
        json_data = json.load(fp)
    return json_data

def save_json(json_file, json_map):
    with open(json_file, 'w', encoding='utf-8') as fp:
        json.dump(json_map, fp, ensure_ascii=False, indent=4)