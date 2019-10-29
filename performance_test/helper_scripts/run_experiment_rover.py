import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from shlex import shlex

import csv
from StringIO import StringIO

import pandas as pandas


def filename_stem():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_experiment(
    topic='Array16',
    rate=100,
    rmw='',
    num_subs=10,
    num_pubs=10,
    timeout=30,
    output_directory='./log',
    filename=None
):
    filename = Path(output_directory) / (
        filename or (filename_stem() + '.json'))

    new_env = dict(os.environ)
    new_env['RMW_IMPLEMENTATION'] = rmw

    test_record = {
        'cmd': ['perf_test',
            '--topic', str(topic),
            '--rate', str(rate),
            '--num_subs', str(num_subs),
            '--num_pubs', str(num_pubs)
        ],
        'timeout': timeout,
        'env': new_env,
    }

    try:
        res = subprocess.run(
            test_record['cmd'], timeout=timeout + 0.5,
            capture_output=True, env=new_env)
        test_record['stdout'] = res.stdout
        test_record['stderr'] = res.stderr
    except subprocess.TimeoutExpired as e:
        test_record['stdout'] = e.stdout
        test_record['stderr'] = e.stderr

    with open(filename, 'w') as f:
        json.dump(test_record, f)


def merge_data(files):
    all_data = []
    for file in files:
        text = Path(file).read_text()
        header_part, data_part = text.split('\n---EXPERIMENT-START---\n', 2)
        data_io = StringIO(data_part)
        df = pandas.read_csv(data_io)
        header_data = dict()
        for line in header_part.splitlines():
            for k, v in line.split(':', 2):
                header_data[k] = v
        df.assign(**header_data)
        all_data.append(df)
    return pandas.concat(all_data)


def plot_data(df):
    grouped = df.group_by('rmw').mean()
    grouped.plot.scatter(x='num_pubs', y='latency_mean')

    pass


def run_experiments():
    pass


def plot_results():
    pass
