import sys
sys.path.append('/tmp/share/repos/adrian/Spacenet8/baseline')
import cache_train
import experiments.pretrain.run as r

def escape(s):
    return s.replace('_', '\\_')

def create_table(header, rows, caption=None):
    lines = []
    lines.append('\\begin{table}')
    lines.append(''.join(['    \\begin{tabular}{','l'*len(header), '}']))

    def create_row(row):
        parts = ['        ']
        for i, cell in enumerate(row):
            if type(cell) == float:
                cell = '%.2f' % (cell, )
            parts.append(escape(str(cell)))
            if i == len(row) - 1:
                parts.append(' \\\\')
            else:
                parts.append(' & ')
        return ''.join(parts)

    lines.append(create_row(header))
    lines.append('        \\hline')
    for row in rows:
        lines.append(create_row(row))
    lines.append('    \\end{tabular}')
    if caption:
        lines.append('    \\caption{' + escape(caption) + '}')
    lines.append('\\end{table}')
    return  '\n'.join(lines)

def create_flood_eval_table(metrics, caption):
    header = ['', 'Non-Flooded Building', 'Flooded Building', 'Non-flooded Road',
    'Flooded Road']
    metrics_by_class = metrics['flood eval']['metrics_by_class']
    def get_metric(metric):
        return [
            metrics_by_class['non-flooded building'][metric],
            metrics_by_class['flooded building'][metric],
            metrics_by_class['non-flooded road'][metric],
            metrics_by_class['flooded road'][metric],
        ]
    rows = [
        [metric] + get_metric(metric) for metric in [
            'precision', 'recall', 'f1', 'iou'
        ]
    ]
    return create_table(header, rows, caption)

def create_foundation_evals_table(foundation_metrics, caption):
    header = ['Model', 'Pre-trained', 'Building IoU', 'Road IoU']
    rows = [
        [
            metrics['foundation eval']['model_name'],
            'yes' if metrics['foundation training']['from_pretrained'] else 'no',
            metrics['foundation eval']['metrics_by_class']['building']['iou'],
            metrics['foundation eval']['metrics_by_class']['road']['iou'],
        ] for metrics in foundation_metrics
    ]
    return create_table(header, rows, caption)

def gen_tables():
    cache = cache_train.RunCache(r.SAVE_DIR)

    foundation_metrics = [
       cache.get_run_metrics(run_config) for run_config in r.p1_runs]
    flood_metrics = [
       cache.get_run_metrics(run_config) for run_config in r.p2_runs]
    tables = []
    for metrics in flood_metrics:
        caption = ' '.join([
            'Metrics for', 
            'pretrained' if metrics['flood training']['from_pretrained'] else 'not-pretrained',
            metrics['flood eval']['model_name']])
        tables.append(create_flood_eval_table(metrics, caption))
    tables.append(create_foundation_evals_table(foundation_metrics,
        'Metrics for Foundation Features Models'))
    print('\n\n'.join(tables))

if __name__ == '__main__':
    gen_tables()
