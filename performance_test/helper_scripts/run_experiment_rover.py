import json
import os
import subprocess
import uuid
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from shlex import shlex

import csv

from pandas import DataFrame
from StringIO import StringIO

import pandas as pandas


def filename_stem():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_experiment(
        topic='Array1k',
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


C = namedtuple('Column', ['name', 'parser'])

experiment_columns = {
    'Experiment id': C('experiment_id', str),
    'Performance Test Version': C('gitversion', str),
    'Logfile name': C('logfile', str),
    'Communication mean': C('communication', str),
    'RMW Implementation': C('rmw', str),
    'QOS': C('qos', str),
    'Publishing rate': C('rate', int),
    'Topic name': C('topic', str),
    'Maximum runtime (sec)': C('runtime', int),
    'Number of publishers': C('num_pub', int),
    'Number of subscribers': C('num_sub', int),
    'Memory check enabled': C('memcheck', bool),
    'Use ros SHM': C('use_ros_shm', bool),
    'Use single participant': C('single_participant', bool),
    'Not using waitset': C('no_waitset', bool),
    'Not using Connext DDS Micro INTRA': C('disable_micro_intra', bool),
    'With security': C('use_security', bool),
    'Roundtrip Mode': C('Roundtrip Mode', str),
}

metric_columns = {
    'T_experiment': C('experiment_time', float),
    'T_loop': C('d_time', float),
    'received': C('received', int),
    'sent': C('sent', int),
    'lost': C('lost', int),
    'latency_mean(ms)': C('latency_mean_ms', float),
    'cpu_usage (%)': C('cpu_usage', lambda x: float(x) / 100.0),
    'ru_maxrss': C('ru_maxrss', int),
}


def merge_data(files):
    all_experiments = []
    all_metrics = []
    for file in files:
        text = Path(file).read_text()
        experiment_text, metrics_test = text.split('\n---EXPERIMENT-START---\n', 2)

        experiment_dict = {}
        for line in experiment_text.splitlines():
            for k, v in line.split(':', 2):
                k = k.strip()
                v = v.strip()
                column_def = experiment_columns.get(k)
                if column_def is None:
                    experiment_dict[k] = v
                else:
                    experiment_dict[column_def.name] = column_def.parser(v)
        all_experiments.append(experiment_dict)

        experiment_id = experiment_dict['experiment_id']

        data_io = StringIO(metrics_test)
        df = pandas.read_csv(data_io)
        df.assign(experiment_id=experiment_id)
        header_data = dict()
        for line in experiment_text.splitlines():
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
