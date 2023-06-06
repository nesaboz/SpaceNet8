import argparse
import end2end
import json
import os
import shutil

'''
Directory Structure
<save_dir>/
  cache.json
  flood-<model-0>-0/ - training and evaluation output for flood network of type
                       <model>. e.g. flood-resnet34_siamese-0
  flood-<model-1>-1/
  flood-<model-2>-2/
  ...
  foundation-<model-0>-0/ - training and evaluation output for foundation
                            network of type <model>. e.g. foundation-resnet34-0
  foundation-<model-1>-1/
  foundation-<model-2>-2/
  ...
  metrics.json - metrics for all the models
'''

class RunCache:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.cache_path = os.path.join(base_dir, 'cache.json')
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        self.directory_counter = self.cache.get('directory_counter', 0)

    def get_run_metrics(self, run_config):
        return self.cache_path.get(run_confg.run_id())

    def save_run_metrics(self, run_config, metrics):
        run_id = run_config.run_id()
        if run_id in self.cache:
            print('WARNING: overwriting existing metrics for run id %r' % run_id)
        self.cache[run_id] = metrics
        self.save()

    def new_run_directory(self, run_config, prefix=''):
        if prefix:
            prefix = prefix + '-'
        self.directory_counter += 1
        # If training crashes we don't want to re-used the messed up directory
        # number
        self.save()
        return os.path.join(self.base_dir, f'{prefix}{self.directory_counter}')

    def save(self):
        os.makedirs(os.path.join(self.base_dir, 'backups'), exist_ok=True)
        backup_path = os.path.join(self.base_dir, 'backups', f'cache{self.directory_counter}.json')
        if os.path.exists(self.cache_path):
            shutil.copy(self.cache_path, backup_path)
        self.cache['directory_counter'] = self.directory_counter
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=4)

class RunConfig:
    def __init__(self, train_csv, val_csv, model_name,
            from_pretrained, lr, batch_size, n_epochs, flood):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.model_name = model_name
        self.from_pretrained = from_pretrained
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.flood = flood

    def run_id(self):
        return '#'.join([
            self.train_csv,
            self.val_csv,
            self.model_name,
            'pretrained' if self.from_pretrained else 'not-pretrained',
            str(self.lr),
            str(self.batch_size),
            str(self.n_epochs),
            'flood' if self.flood else 'foundation'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",
                        type=str,
                        required=True)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)

    parser.add_argument("--train_csv",
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument("--val_csv",
                        type=str,
                        nargs='+',
                        required=True)

    parser.add_argument("--foundation_model_names",
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of foundation models to train')
    parser.add_argument("--foundation_model_from_pretrained",
                        type=str,
                        nargs='+',
                        choices=['true', 'false'],
                        help='Whether to initialize from pretrained weights for each model')
    parser.add_argument("--foundation_lr",
                        type=float,
                        nargs='+',
                        help='List of learning rates for each model')
    parser.add_argument("--foundation_batch_size",
                        type=int,
                        nargs='+',
                        help='List of batch sizes for each model')
    parser.add_argument("--foundation_n_epochs",
                        type=int,
                        nargs='+',
                        help='List of batch sizes for each model')

    parser.add_argument("--flood_model_names",
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of flood models to train')
    parser.add_argument("--flood_model_from_pretrained",
                        type=str,
                        nargs='+',
                        choices=['true', 'false'],
                        help='Whether to initialize from pretrained weights for each model')
    parser.add_argument("--flood_lr",
                        type=float,
                        nargs='+',
                        help='List of learning rates for each model')
    parser.add_argument("--flood_batch_size",
                        type=int,
                        nargs='+',
                        help='List of batch sizes for each model')
    parser.add_argument("--flood_n_epochs",
                        type=int,
                        nargs='+',
                        help='List of batch sizes for each model')
    return parser.parse_args()

def create_run_configs(args):
    def get(model_index, num_models, values):
        if len(values) == 1:
            return values[0]
        if len(values) == num_models:
            return values[model_index]
        raise ValueError('Must either specify 1 value or <num models> values')

    n_foundation = len(args.foundation_model_names)
    foundation_runs = [
        RunConfig(
        train_csv=get(i, n_foundation, args.train_csv),
        val_csv=get(i, n_foundation, args.val_csv),
        model_name=model_name,
        from_pretrained=get(i, n_foundation, args.foundation_model_from_pretrained) == 'true',
        lr=get(i, n_foundation, args.foundation_lr),
        batch_size=get(i, n_foundation, args.foundation_batch_size),
        n_epochs=get(i, n_foundation, args.foundation_n_epochs),
        flood=False)
        for i, model_name in enumerate(args.foundation_model_names)
    ]

    n_flood = len(args.flood_model_names)
    flood_runs = [
        RunConfig(
        train_csv=get(i, n_foundation, args.train_csv),
        val_csv=get(i, n_foundation, args.val_csv),
        model_name=model_name,
        from_pretrained=get(i, n_flood, args.flood_model_from_pretrained) == 'true',
        lr=get(i, n_flood, args.flood_lr),
        batch_size=get(i, n_flood, args.flood_batch_size),
        n_epochs=get(i, n_flood, args.flood_n_epochs),
        flood=True)
        for i, model_name in enumerate(args.flood_model_names)
    ]
    return foundation_runs + flood_runs

def train_models(save_dir, runs=[], gpu=0):
    print('Training %d models...' % (len(runs), ))
    os.makedirs(save_dir, exist_ok=True)
    cache = RunCache(save_dir)
    for r in runs:
        print('Starting', r.run_id(), '...')
        if r.flood:
            run_metrics = end2end.run(
                save_dir=cache.new_run_directory(r, 'flood'),
                train_csv=r.train_csv,
                val_csv=r.val_csv,
                gpu=gpu,
                flood_model_name=r.model_name,
                flood_lr=r.lr,
                flood_batch_size=r.batch_size,
                flood_n_epochs=r.n_epochs,
                flood_model_args={
                    'from_pretrained':r.from_pretrained
                })
        else:
            run_metrics = end2end.run(
                save_dir=cache.new_run_directory(r, 'foundation'),
                train_csv=r.train_csv,
                val_csv=r.val_csv,
                gpu=gpu,
                foundation_model_name=r.model_name,
                foundation_lr=r.lr,
                foundation_batch_size=r.batch_size,
                foundation_n_epochs=r.n_epochs,
                foundation_model_args={
                    'from_pretrained':r.from_pretrained
                })
        cache.save_run_metrics(r, end2end.values_to_json_obj(run_metrics))
    return cache

if __name__ == '__main__':
    args = parse_args()
    runs = create_run_configs(args)
    train_models(args.save_dir, runs, args.gpu)
