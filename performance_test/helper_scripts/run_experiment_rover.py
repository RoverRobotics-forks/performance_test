import json
import os
import platform
import shutil
import subprocess
import sys
from collections import namedtuple
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import cpuinfo  # from py-cpuinfo
import pandas as pandas
from pandas import DataFrame


def filename_stem():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    search_paths = []
    for p in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep):
        if not p: continue
        p2 = os.path.join(p, 'lib/performance_test')
        if p in search_paths: continue
        search_paths.append(p2)
    for p in os.environ.get('PATH', '').split(os.pathsep):
        if p and p not in search_paths:
            search_paths.append(p)
    full_search_path = ':'.join(search_paths)

    for rmw in ('rmw_cyclonedds_cpp', 'rmw_fastrtps_cpp'):
        for topic in (
                'Array1k',  # 'Array4k',
                'Array16k',  # 'Array32k',
                'Array60k',  # 'Array1m',
                'Array2m'
        ):
            run_experiment(
                exe_path=full_search_path,
                rmw=rmw,
                reliable=True,
                num_subs=16,
                topic=topic,
                max_runtime=30,
                data_dir='data',
                data_file_prefix=rmw + '_' + topic + '_',
            )

    plot_data('data')


def run_experiment(
        exe_path=None,
        topic='Array1k',
        rate=100,
        rmw='',
        num_subs=10,
        num_pubs=1,
        max_runtime=30,
        dds_domain_id=77,
        history_depth: Optional[int] = 100,
        reliable=False,
        data_dir='data',
        data_file_prefix=None,
):
    new_env = dict(os.environ)
    new_env.update(
        RMW_IMPLEMENTATION=rmw,
    )
    new_env['RMW_IMPLEMENTATION'] = rmw
    cmd = [
        shutil.which('perf_test', path=exe_path),
        '--communication', 'ROS2',
        '--topic', str(topic),
        '--rate', str(rate),
        '--num_sub_threads', str(num_subs),
        '--num_pub_threads', str(num_pubs),
        '--max_runtime', str(max_runtime),
        *(['--reliable'] if reliable else []),
        *(['--keep_last', '--history_depth',
           str(history_depth)] if history_depth is not None else []),
        '--dds_domain_id', str(dds_domain_id),
    ]

    try:
        res = subprocess.run(cmd, timeout=max_runtime + 0.5,
                             env=new_env,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             encoding='utf-8')
        stdout = res.stdout
        stderr = res.stderr

    except subprocess.TimeoutExpired as e:
        stdout = e.stdout
        stderr = e.stderr

    if stderr:
        print(stderr, file=sys.stderr)
    if not stdout:
        print('no stdout from perf_test', file=sys.stderr)

    experiment_data = extract_experiment_data(stdout)
    experiment_id = experiment_data['experiment_id']
    experiment_data.update({
        'cmd': subprocess.list2cmdline(cmd),
        'timeout': max_runtime,
        'env': repr(new_env),
        'cpu': cpuinfo.get_cpu_info()['brand'],
        'host': platform.node(),
        'os': platform.platform(),
    })

    experiment_metrics = extract_metric_data(stdout)
    experiment_metrics['experiment_id'] = experiment_id

    data_content = {'experiment': experiment_data, 'metrics': experiment_metrics.to_dict()}
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    data_file_name = (data_file_prefix or '') + filename_stem() + '.json'
    with Path(data_dir, data_file_name).open('w') as f:
        json.dump(data_content, f, indent=2)


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
    # todo: the rest of the columns
}

import warnings


def extract_experiment_data(text: str):
    text = text.split('---EXPERIMENT-START---')[0]
    result = dict()
    for line in text.splitlines():
        try:
            k, v = line.split(':', 1)
        except Exception as e:
            warnings.warn('failed to parse line: ' + line + str(e))
            continue
        k = k.strip()
        v = v.strip()
        column_def = experiment_columns.get(k)
        if column_def is None:
            result[k] = v
        else:
            result[column_def.name] = column_def.parser(v)
    return result


def extract_metric_data(text: str):
    text = text.split('---EXPERIMENT-START---')[-1]
    text = text.split('Maximum runtime reached. Exiting.')[0]
    data_io = StringIO(text)
    return pandas.read_csv(data_io, sep=r'\s*,\s*', engine='python')


import matplotlib.pyplot as plt

size_by_topic = {
    'Array1k': 1 << 10,
    'Array4k': 4 << 10,
    'Array16k': 16 << 10,
    'Array32k': 32 << 10,
    'Array60k': 60 << 10,
    'Array1m': 1 << 20,
    'Array2m': 2 << 20,
}


def plot_data(data_dir):
    all_experiments = []
    all_metrics = []
    for file in Path(data_dir).glob('*.json'):
        with file.open('r') as f:
            dobj = json.load(f)
        all_experiments.append(dobj['experiment'])
        all_metrics.append(DataFrame.from_dict(dobj['metrics']))
    e = DataFrame.from_records(all_experiments, index=['experiment_id'])
    e['message_size'] = e['topic'].replace(size_by_topic)

    m = pandas.concat(all_metrics)
    # The first few seconds have suspiciously low CPU usage;
    # I suspect a measuring artifact
    m = m.loc[m['T_experiment'] > 3]

    all_data = m.join(e, on=['experiment_id'])
    all_data.sort_values(by='message_size', inplace=True)
    all_data['all'] = ''

    import seaborn as sns
    plt.figure()
    ax = sns.violinplot(x='topic', y='latency_mean (ms)', hue='rmw', data=all_data, split=True)
    ax.set(xlabel='Topic', ylabel='Mean Latency (ms) - lower is better', yscale='log')
    # plt.show()
    plt.savefig('latency.png')

    plt.figure()
    ax = sns.violinplot(x='topic', y='cpu_usage (%)', hue='rmw', data=all_data, split=True)
    ax.set(xlabel='Topic', ylabel='CPU usage (%) - lower is better', yscale='linear')
    # plt.show()
    plt.savefig('cpu.png')

    plt.figure()
    all_data['ru_maxrss_mb'] = all_data['ru_maxrss'] / 1000
    ax = sns.barplot(x='topic', y='ru_maxrss_mb', hue='rmw', data=all_data, ci=None)
    ax.set(xlabel='Topic', ylabel='RAM Usage (MB) - lower is better', yscale='linear')
    # plt.show()
    plt.savefig('ram_usage.png')


if __name__ == '__main__':
    main()
