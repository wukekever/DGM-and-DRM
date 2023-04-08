import yaml
from yaml.loader import Loader

def get_yaml(yaml_file):
    # read data by open yaml file
    file = open(yaml_file, 'r', encoding = "utf-8")
    file_data = file.read()
    file.close()
    # str -> dic or list
    dic = yaml.load(file_data, Loader = yaml.FullLoader)    
    return dic