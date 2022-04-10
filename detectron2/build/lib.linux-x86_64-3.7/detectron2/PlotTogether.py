# ---------------------------------------------------------------------
# https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
# ---------------------------------------------------------------------
import json
import matplotlib.pyplot as plt

experiment_folder = '/content/drive/.shortcut-targets-by-id/18zBrqNwnAWKJDMVcLIF8bsGLyzer_SoE/IW/code_in_checkpoints'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

plt.plot(
    [x['iteration'] for x in experiment_metrics], 
    [x['total_loss'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()