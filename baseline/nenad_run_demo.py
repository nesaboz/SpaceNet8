
from pathlib import Path
from datetime import datetime
from end2end import run
import os


train_csv="/tmp/share/data/spacenet8/sn8_data_train.csv"
val_csv="/tmp/share/data/spacenet8/sn8_data_val.csv"

# train_csv="/tmp/share/data/spacenet8/adrs-small-train.csv"
# val_csv="/tmp/share/data/spacenet8/adrs-small-val.csv"

run_root = Path('/tmp/share/runs/spacenet8/nenad')

now = datetime.now() 
tag = '_segformer_b2'
save_dir = os.path.join(run_root, now.strftime("%Y-%m-%d-%H-%M") + tag)

run(
    save_dir=save_dir,
    train_csv=train_csv,
    val_csv=val_csv,
    foundation_model_name='segformer_b2',   # resnet34
    foundation_lr=0.0001,
    foundation_batch_size=2,
    foundation_n_epochs=10,
    flood_model_name='segformer_b2_siamese',  #  # resnet34_siamese
    flood_lr=0.0001,
    flood_batch_size=1,
    flood_n_epochs=10
)

# now = datetime.now() 
# tag = '_resnet34'
# save_dir = os.path.join(run_root, now.strftime("%Y-%m-%d-%H-%M") + tag)

# run(
#     save_dir=save_dir,
#     train_csv=train_csv,
#     val_csv=val_csv,
#     foundation_model_name='resnet34',   # resnet34
#     foundation_lr=0.0001,
#     foundation_batch_size=4,
#     foundation_n_epochs=10,
#     flood_model_name='resnet34_siamese',  #  # resnet34_siamese
#     flood_lr=0.0001,
#     flood_batch_size=2,
#     flood_n_epochs=10
# )

# # extra code to run evaluation only

# from foundation_eval import foundation_eval
# from flood_eval import flood_eval

# def eval_wrapper(folder, model_name, in_csv, eval_func):
#     def get_params(folder):
#         model_path = os.path.join(folder, 'best_model.pth')
#         save_fig_dir = os.path.join(folder, 'pngs')
#         save_pred_dir = os.path.join(folder, 'tiffs')
#         return model_path, save_fig_dir, save_pred_dir
    
#     model_path, save_fig_dir, save_pred_dir = get_params(folder)
#     eval_func(model_path=model_path, in_csv=in_csv, save_fig_dir=save_fig_dir, save_preds_dir=save_pred_dir, model_name=model_name, gpu=0, create_folders=True)
    
# eval_wrapper(
#     folder=run_root/'foundation/02-06-2023-05-07', 
#     flood_model_name='segformer_b1',
#     in_csv=val_csv,
#     eval_func=foundation_eval
#     )