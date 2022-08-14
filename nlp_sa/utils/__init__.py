import yaml
from argparse import Namespace


def get_config(config_file: str) -> Namespace:
    """
    Loads paramerts from a yaml configuration file in a
    Namespce object.
    Args:
      config_file: A path to a yaml configuration file.
    Returns:
      A Namespace object that contans configurations referenced
      in the program.
    """
    stream = open(config_file, 'r')
    config_dict = yaml.load(stream, yaml.SafeLoader)

    for parameter, value in config_dict.items():
        print("{0:30} {1}".format(parameter, value))

    config = Namespace(**config_dict)

    return config


def get_label_list(labels):
    """
    Function to get list of labels
    """
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list