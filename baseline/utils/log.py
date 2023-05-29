import sys

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
