import argparse
import end2end
import json
import os
import train_flood
import train_foundation_features
from utils.log import dump_command_line_args

'''
Directory Structure
<save_dir>/
  args.txt - command line args for this script
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

class RunConfig:
    def __init__(self, train_csv, val_csv, save_dir, gpu, model_name,
            from_pretrained, lr, batch_size, n_epochs, run_label):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.save_dir = save_dir
        self.gpu = gpu
        self.model_name = model_name
        self.from_pretrained = from_pretrained
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.run_label = run_label

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True,
                        default="/tmp/share/data/spacenet8/sn8_data_train.csv")
    parser.add_argument("--val_csv",
                        type=str,
                        required=True,
                        default="/tmp/share/data/spacenet8/sn8_data_val.csv")
    parser.add_argument("--save_dir",
                        type=str,
                        required=True)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)

    parser.add_argument("--foundation_model_names",
                        type=str,
                        nargs='+',
                        default=list(train_foundation_features.models.keys()),
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
                        default=list(train_flood.models.keys()),
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

    # Include the run index in the directory path and run label in case there
    # are multiple runs with the same model name.
    n_foundation = len(args.foundation_model_names)
    foundation_runs = [
        RunConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        save_dir=os.path.join(args.save_dir, f'foundation-{model_name}-{i}'),
        gpu=args.gpu,
        model_name=model_name,
        from_pretrained=get(i, n_foundation, args.foundation_model_from_pretrained) == 'true',
        lr=get(i, n_foundation, args.foundation_lr),
        batch_size=get(i, n_foundation, args.foundation_batch_size),
        n_epochs=get(i, n_foundation, args.foundation_n_epochs),
        run_label=f'foundation-{model_name}-{i}') for i, model_name in enumerate(args.foundation_model_names)
    ]

    n_flood = len(args.flood_model_names)
    flood_runs = [
        RunConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        save_dir=os.path.join(args.save_dir, f'flood-{model_name}-{i}'),
        gpu=args.gpu,
        model_name=model_name,
        from_pretrained=get(i, n_flood, args.flood_model_from_pretrained) == 'true',
        lr=get(i, n_flood, args.flood_lr),
        batch_size=get(i, n_flood, args.flood_batch_size),
        n_epochs=get(i, n_flood, args.flood_n_epochs),
        run_label=f'flood-{model_name}-{i}') for i, model_name in enumerate(args.flood_model_names)
    ]
    return foundation_runs, flood_runs

def run_with_error_tolerance(f):
    return f()
    try:
        return f()
    except Exception as e:
        print('ERROR:', e)
        return {'error': str(e)}

def train_models(save_dir, foundation_runs=[], flood_runs=[]):
    print('Training %d foundation and %d flood models...' %
        (len(foundation_runs), len(flood_runs)))
    os.makedirs(save_dir, exist_ok=True)
    metrics = {}
    for r in foundation_runs:
        print('Starting', r.run_label, '...')
        run_metrics = run_with_error_tolerance(lambda: end2end.run(
            save_dir=r.save_dir,
            train_csv=r.train_csv,
            val_csv=r.val_csv,
            gpu=r.gpu,
            foundation_model_name=r.model_name,
            foundation_lr=r.lr,
            foundation_batch_size=r.batch_size,
            foundation_n_epochs=r.n_epochs,
            foundation_model_args={
                'from_pretrained':r.from_pretrained
            }))
        metrics[r.run_label] = end2end.values_to_json_obj(run_metrics)
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    for r in flood_runs:
        print('Starting', r.run_label, '...')
        run_metrics = run_with_error_tolerance(lambda: end2end.run(
            save_dir=r.save_dir,
            train_csv=r.train_csv,
            val_csv=r.val_csv,
            gpu=r.gpu,
            flood_model_name=r.model_name,
            flood_lr=r.lr,
            flood_batch_size=r.batch_size,
            flood_n_epochs=r.n_epochs,
            flood_model_args={
                'from_pretrained':r.from_pretrained
            }))
        metrics[r.run_label] = end2end.values_to_json_obj(run_metrics)
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    return metrics

if __name__ == '__main__':
    args = parse_args()
    dump_command_line_args(os.path.join(args.save_dir, 'args.txt'))
    foundation_runs, flood_runs = create_run_configs(args)
    train_models(args.save_dir, foundation_runs, flood_runs)
