import os
import subprocess
from shlex import shlex


def run_experiment(
        topic='Array16',
        rate=100,
        rmw='',
        num_subs=10,
        num_pubs=10,
        timeout=30,
):
    topic_args = ['--topic {}'.format(t) for t in topics]
    rate_args = ['--rate {}'.format(r) for r in rates]
        
    new_env = dict(os.environ)
    new_env['RMW_IMPLEMENTATION'] = rmw
    
    try:
      res = subprocess.run(['perf_test',
                    '--topic','topic',
                    '--rate','rate',
                    '--num_subs' str(num_subs),
                   ], timeout=timeout+0.5, capture_output=True, env=new_env)
      o= res.stdout
     except TimeoutExpired as e:
      o = e.stdout
     rawdata = {
             "kwargs": kwargs,
             "stdout": o,
     }


def run_experiments():
    pass


def plot_results():
    pass
