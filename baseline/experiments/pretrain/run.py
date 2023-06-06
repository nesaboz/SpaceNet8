import sys
sys.path.append('/tmp/share/repos/adrian/Spacenet8/baseline')
import cache_train

SAVE_DIR="/tmp/share/runs/adrs/all-experiments"

batch_size_by_model = {
    'resnet34': 8,
    'segformer_b0': 7,
    'resnet34_siamese': 6,
    'segformer_b0_siamese': 4,
    'segformer_b1': 6,
    'segformer_b1_siamese': 4,
}

is_flood_by_model = {
    'resnet34': False,
    'segformer_b0': False,
    'resnet34_siamese': True,
    'segformer_b0_siamese': True,
    'segformer_b1': False,
    'segformer_b1_siamese': True,
}

def create_config(model_name, from_pretrained):
    return cache_train.RunConfig(
        train_csv='/tmp/share/data/spacenet8/sn8_data_train.csv',
        val_csv='/tmp/share/data/spacenet8/sn8_data_val.csv',
        model_name=model_name,
        from_pretrained=from_pretrained,
        lr=0.0001,
        batch_size=batch_size_by_model[model_name],
        n_epochs=10,
        flood=is_flood_by_model[model_name])

p1_runs = [
    create_config('resnet34', from_pretrained=True),
    create_config('resnet34', from_pretrained=False),
    create_config('segformer_b0', from_pretrained=True),
    create_config('segformer_b0', from_pretrained=False),
]

p2_runs = [
    create_config('resnet34_siamese', from_pretrained=True),
    create_config('resnet34_siamese', from_pretrained=False),
    create_config('segformer_b0_siamese', from_pretrained=True),
    create_config('segformer_b0_siamese', from_pretrained=False),
]

def run():
    print('Training p1s...')
    cache_train.train_models(SAVE_DIR, p1_runs, gpu=0)
    print('Training p2s...')
    cache_train.train_models(SAVE_DIR, p2_runs, gpu=0)

if __name__ == '__main__':
    run()
