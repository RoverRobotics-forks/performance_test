# Copyright 2019 Apex.AI, Inc.
# All rights reserved.

import argparse
import collections
import itertools
import json
import sys


# Fields that are required to be in the JSON file
_field_names = ['topics', 'rates', 'num_subs', 'reliability', 'durability', 'keep_last']
_optional_field_names = ['history_depth']


class _PerfConfig():

    def __init__(self,
                 topic,
                 rate,
                 num_subs,
                 reliability,
                 durability,
                 keep_last,
                 history_depth=None):
        self.topic = topic
        self.rate = rate
        self.num_subs = num_subs
        self.reliability = reliability
        self.durability = durability
        self.keep_last = keep_last
        self.history_depth = history_depth

    def command_line(self):
        cfg_str = "--topic {} --rate {} --num_sub_threads {} {} {} {}".format(
            self.topic,
            self.rate,
            self.num_subs,
            self.reliability,
            self.durability,
            self.keep_last
        )

        # Now for optional fields:
        if self.history_depth is not None:
            cfg_str += " --history_depth {}".format(self.history_depth)

        return cfg_str


def make_args():
    parser = argparse.ArgumentParser(
        description='Generates command-line arguments to performance_test from a config file'
    )

    parser.add_argument('config_file', action='store',
                        help='The path to a config file')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        cfg = json.load(f)

    try:
        validate_config(cfg)
    except ValueError as e:
        sys.exit(e)

    expanded_configurations = generate_config(cfg)

    for configuration in expanded_configurations:
        print(configuration.command_line())


def config_is_excluded(config, exclude_dict_list):
    """If all of the values in the exclude_dict match the config, it's excluded."""

    if not exclude_dict_list:
        return False

    def dict_excludes_config(exclude_dict):
        for key, value in exclude_dict.items():
            if value != getattr(config, key):
                return False

        return True

    return any(dict_excludes_config(d) for d in exclude_dict_list)


def generate_config(config_dict):
    """
    Generate command-line arguments for the performance_test tool.

    Returns a list of NamedTuple objects
    """

    config_dict['matrix'] = collections.defaultdict(lambda: [None], **config_dict['matrix'])

    values = (config_dict['matrix'][key] for key in _field_names + _optional_field_names)

    combos = itertools.product(*values)

    # This is a generator of 'PerfConfig' named tuple objects.  It still needs to be filtered
    matrix = (_PerfConfig(*c) for c in combos)

    if 'exclude' not in config_dict:
        config_dict['exclude'] = []

    return [cfg for cfg in matrix if not config_is_excluded(cfg, config_dict['exclude'])]


def validate_config(config_dict):
    """
    Check that a config dictionary contains the necessary information for generate_config.

    This method will raise a ValueError with a meaningful description of the problem if
    validation fails
    """

    expected_keys = _field_names

    for key in expected_keys:
        if key not in config_dict['matrix']:
            raise ValueError("Missing key '{}' in configuration".format(key))

    possible_keys = _field_names + _optional_field_names
    for key in config_dict['matrix'].keys():
        if key not in possible_keys:
            raise ValueError("Unexpected extra key '{}' found in configuration".format(key))
