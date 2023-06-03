import sys
import psutil
import time
import torch
import torch.cuda
import inspect
import json
import os

class ResourceMetricsMixin:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = None

    def start(self):
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()

    def end(self):
        self.peak_memory = torch.cuda.max_memory_allocated()
        self.end_time = time.time()

    def resource_metrics(self):
        return {
            'peak_memory':self.peak_memory,
            'runtime': self.end_time - self.start_time
        }

class TrainingMetrics(ResourceMetricsMixin):
    def __init__(self):
        self.best_loss = float('inf')
        self.epochs = []

    # TODO: track loss over each iteration

    def add_epoch(self, metrics):
        self.epochs.append(metrics)
        loss = metrics['val_tot_loss']
        if loss < self.best_loss:
            self.best_loss = loss

    def to_json_object(self):
        metrics = self.resource_metrics()
        metrics['best_loss'] = self.best_loss
        metrics['epochs'] = self.epochs
        return metrics

class EvalMetrics(ResourceMetricsMixin):
    def __init__(self):
        self.metrics_by_class = {}

    def add_class_metrics(self, class_label, metrics):
        self.metrics_by_class[class_label] = metrics

    def to_json_object(self):
        metrics = self.resource_metrics()
        metrics['metrics_by_class'] = self.metrics_by_class
        return metrics


debug = False
def debug_msg(msg):
    'Prints a message if debuggign is enabled'
    if debug:
        print(msg)

def log_var_details(name, var):
    'Logs the type and shape of a variable'
    t = type(var)
    s = getattr(var, 'shape', None)
    dtype = getattr(var, 'dtype', None)
    debug_msg(f'Variable: {name}, Type: {t}, Shape: {s}, Dtype: {dtype}')

def dump_command_line_args(path):
    'Saves the command line arguments to path to help reproduce training runs.'
    with open(path, 'w') as f:
        f.write(str(sys.argv))

def dump_to_json(save_dir, params):
    with open(os.path.join(save_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=4)  # `indent` writes each new input on a new line
        
def load_from_json(input_path):
    with input_path.open() as f:
        return json.load(f)
        
def get_fcn_params(frame):
    args, _, _, values = inspect.getargvalues(frame)
    return {k: values[k] for k in args}
    

def print_gpu_memory(verbose=False):
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the default CUDA device
        device = torch.cuda.current_device()

        # Get the total memory and currently allocated memory on the device
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)

        # Convert bytes to megabytes
        total_memory_mb = total_memory / 1024**2
        allocated_memory_mb = allocated_memory / 1024**2

        if verbose:
            # Print the memory information
            print(f"Total GPU memory: {total_memory_mb:.2f} MB")
            print(f"Allocated GPU memory: {allocated_memory_mb:.2f} MB")
        else:
            print(f"Percentage of used GPU memory: {allocated_memory_mb/total_memory_mb}%")
    else:
        print("CUDA is not available")


def print_cpu_memory(verbose=True):

    # Get the current memory usage
    memory_info = psutil.virtual_memory()

    # Extract the memory information
    total_memory = memory_info.total

    available_memory = memory_info.available
    used_memory = memory_info.used
    percent_memory = memory_info.percent

    # Convert bytes to megabytes
    total_memory_mb = total_memory / 1024**2
    available_memory_mb = available_memory / 1024**2
    used_memory_mb = used_memory / 1024**2

    if verbose:
        # Print the memory information
        print(f"Total CPU memory: {total_memory_mb:.2f} MB")
        print(f"Available CPU memory: {available_memory_mb:.2f} MB")
        print(f"Used CPU memory: {used_memory_mb:.2f} MB")
        print(f"Percentage of used CPU memory: {percent_memory}%")
    else:
        print(f"Percentage of used CPU memory: {percent_memory}%")
