import os
import subprocess
from shlex import shlex


def run_experiment(
        topic='Array16',
        rate=100,
        rmw='',
        num_subs=10,
        num_pubs=10,
):
    topic_args = ['--topic {}'.format(t) for t in topics]
    rate_args = ['--rate {}'.format(r) for r in rates]
    new_env = dict(os.environ)
    new_env['RMW_IMPLEMENTATION'] = rmw

    subprocess.run(['perf_test',
                    '--topic','topic',
                    '--rate','rate',
                    '--num_subs'

                    ], env=new_env)

def run_experiments():
    pass


def plot_results():
    pass