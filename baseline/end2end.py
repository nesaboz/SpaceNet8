import argparse
import json
import matplotlib.pyplot as plt
import os.path

from train_foundation_features import train_foundation
from train_flood import train_flood
from foundation_eval import foundation_eval, get_eval_results_path
from flood_eval import flood_eval
from utils.log import dump_command_line_args

'''
Directory Structure

<save_dir>/
  args.txt - command line args for this script
  metrics.json - training and evaluation metrics (If any part of training or
                 evaluation crashes, these are not saved. That is why there
                 still are log.csv and eval_results.csv files)
  foundation/
    best_model.pth - best foundation model
    log.csv - log of training metrics per-epoch
    model_checkpoint.pth - foundation model checkpoint
    pngs/
    tiffs/
    eval_results.csv - evaluation results
  flood/
    best_model.pth
    log.csv
    model_checkpoint.pth
    pngs/
    tiffs/
    eval_results.csv
  # TODO: plot of training vs validation loss
'''

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

    parser.add_argument("--foundation_model_name",
                         type=str,
                         required=True)
    parser.add_argument("--foundation_lr",
                         type=float,
                        default=0.0001)
    parser.add_argument("--foundation_batch_size",
                         type=int,
                        default=2)
    parser.add_argument("--foundation_n_epochs",
                         type=int,
                         default=50)
    parser.add_argument("--foundation_checkpoint",
                        type=str,
                        default=None)

    parser.add_argument("--flood_model_name",
                         type=str,
                         required=True)
    parser.add_argument("--flood_lr",
                         type=float,
                        default=0.0001)
    parser.add_argument("--flood_batch_size",
                         type=int,
                        default=2)
    parser.add_argument("--flood_n_epochs",
                         type=int,
                         default=50)
    parser.add_argument("--flood_checkpoint",
                        type=str,
                        default=None)
    return parser.parse_args()


def run(
        save_dir,
        train_csv="/tmp/share/data/spacenet8/sn8_data_train.csv",
        val_csv="/tmp/share/data/spacenet8/sn8_data_val.csv",
        gpu=0,
        foundation_model_name=None,
        foundation_lr=0.0001,
        foundation_batch_size=2,
        foundation_n_epochs=50,
        foundation_checkpoint=None,
        foundation_model_args={},
        foundation_kwargs={},
        flood_model_name=None,
        flood_lr=0.0001,
        flood_batch_size=2,
        flood_n_epochs=50,
        flood_checkpoint=None,
        flood_model_args={},
        flood_kwargs={}):
    '''
    Trains and evaluates a foundation features and flood network and returns
    training and evaluation metrics. This function is designed to be used by
    cross-validation routines.
    '''
    metrics = {}
    def save_metrics():
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump({
                key:(val.to_json_object() if hasattr(val, 'to_json_object') else val) for key, val in metrics.items()
                }, f, indent=4)

    if foundation_model_name is not None:
        foundation_dir = os.path.join(save_dir, 'foundation')
        print('Training foundation model...')
        metrics['foundation training'] = train_foundation(
              train_csv=train_csv, 
              val_csv=val_csv,
              save_dir=foundation_dir,
              model_name=foundation_model_name,
              initial_lr=foundation_lr,
              batch_size=foundation_batch_size,
              n_epochs=foundation_n_epochs,
              gpu=gpu,
              checkpoint_path=foundation_checkpoint,
              foundation_model_args=foundation_model_args,
              **foundation_kwargs)
        # Save metrics after training each of the models to see
        # partial results without having to wait for the script to finish.
        save_metrics()

        print('Evaluating foundation model...')
        metrics['foundation eval'] = foundation_eval(
                model_path=os.path.join(foundation_dir, 'best_model.pth'), 
                in_csv=val_csv, 
                save_fig_dir=os.path.join(foundation_dir, 'pngs'),
                save_preds_dir=os.path.join(foundation_dir, 'tiffs'),
                model_name=foundation_model_name)
        save_metrics()

    if flood_model_name is not None:
        flood_dir = os.path.join(save_dir, 'flood')
        print('Training flood model...')
        metrics['flood training'] = train_flood(
              train_csv=train_csv, 
              val_csv=val_csv,
              save_dir=flood_dir,
              model_name=flood_model_name,
              initial_lr=flood_lr,
              batch_size=flood_batch_size,
              n_epochs=flood_n_epochs,
              gpu=gpu,
              checkpoint_path=flood_checkpoint,
              flood_model_args=flood_model_args,
              **flood_kwargs)
        save_metrics()

        print('Evaluating flood model...')
        metrics['flood eval'] = flood_eval(
               model_path=os.path.join(flood_dir, 'best_model.pth'),
               in_csv=val_csv, 
               save_fig_dir=os.path.join(flood_dir, 'pngs'),
               save_preds_dir=os.path.join(flood_dir, 'tiffs'),
               model_name=flood_model_name)
        save_metrics()
    
    return metrics

if __name__ == '__main__':
    args = parse_args()
    dump_command_line_args(os.path.join(args.save_dir, 'args.txt'))
    run(
        save_dir=args.save_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        gpu=args.gpu,
        foundation_model_name=args.foundation_model_name,
        foundation_lr=args.foundation_lr,
        foundation_batch_size=args.foundation_batch_size,
        foundation_n_epochs=args.foundation_n_epochs,
        foundation_checkpoint=args.foundation_checkpoint,
        flood_model_name=args.flood_model_name,
        flood_lr=args.flood_lr,
        flood_batch_size=args.flood_batch_size,
        flood_n_epochs=args.flood_n_epochs,
        flood_checkpoint=args.flood_checkpoint)
    print('Done!')
