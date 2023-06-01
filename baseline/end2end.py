import argparse
import os.path
import matplotlib.pyplot as plt

from train_foundation_features import train_foundation
from train_flood import train_flood
from foundation_eval import foundation_eval, get_eval_results_path
from flood_eval import flood_eval
from utils.log import dump_command_line_args

'''
Directory Structure

<save_dir>/
  args.txt - command line args for this script
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
    args = parser.parse_args()
    return args

def get_eval_dirs(folder):
    model_path = os.path.join(folder, 'best_model.pth')
    model_name = folder.name.split('_lr')[0]
    save_fig_dir = os.path.join(folder, 'pngs')
    save_pred_dir = os.path.join(folder, 'tiffs')
    return model_path, model_name, save_fig_dir, save_pred_dir

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
        flood_model_name=None,
        flood_lr=0.0001,
        flood_batch_size=2,
        flood_n_epochs=50,
        flood_checkpoint=None):

    if foundation_model_name is not None:
        foundation_dir = os.path.join(args.save_dir, 'foundation')
        print('Training foundation model...')
        train_foundation(
              train_csv=args.train_csv, 
              val_csv=args.val_csv,
              save_dir=foundation_dir,
              model_name=args.foundation_model_name,
              initial_lr=args.foundation_lr,
              batch_size=args.foundation_batch_size,
              n_epochs=args.foundation_n_epochs,
              gpu=args.gpu,
              checkpoint_path=args.foundation_checkpoint)
        print('Evaluating foundation model...')
        foundation_eval(model_path=os.path.join(foundation_dir, 'best_model.pth'), 
                in_csv=args.val_csv, 
                save_fig_dir=os.path.join(foundation_dir, 'pngs'),
                save_preds_dir=os.path.join(foundation_dir, 'tiffs'),
                model_name=args.foundation_model_name)

    if flood_model_name is not None:
        flood_dir = os.path.join(args.save_dir, 'flood')
        print('Training flood model...')
        train_flood(
              train_csv=args.train_csv, 
              val_csv=args.val_csv,
              save_dir=flood_dir,
              model_name=args.flood_model_name,
              initial_lr=args.flood_lr,
              batch_size=args.flood_batch_size,
              n_epochs=args.flood_n_epochs,
              gpu=args.gpu,
              checkpoint_path=args.flood_checkpoint)

        print('Evaluating flood model...')
        flood_eval(model_path=os.path.join(flood_dir, 'best_model.pth'),
               in_csv=args.val_csv, 
               save_fig_dir=os.path.join(flood_dir, 'pngs'),
               save_preds_dir=os.path.join(flood_dir, 'tiffs'),
               model_name=args.flood_model_name)
    # TODO: Return a dictionary of metrics to allow this function to be used
    # for hyper-parameter search.
    # TODO: Save additional metrics and generate additional plots

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
