import importlib

# List of required libraries
libraries = [
    'pandas',
    'numpy',
    'tensorflow',
    'matplotlib'
]

# Check if each library is installed, and install if needed
for library in libraries:
    try:
        importlib.import_module(library)
        print(f'{library} is already installed.')
    except ImportError:
        print(f'{library} is not installed. Installing...')
        os.system(f'pip install {library}')

# Additional installation for TensorFlow modules
try:
    importlib.import_module('tensorflow.keras')
    print('tensorflow.keras is already installed.')
except ImportError:
    print('tensorflow.keras is not installed. Installing...')
    os.system('pip install tensorflow')

try:
    importlib.import_module('tensorflow.keras.optimizers.schedules')
    print('tensorflow.keras.optimizers.schedules is already installed.')
except ImportError:
    print('tensorflow.keras.optimizers.schedules is not installed. Installing...')
    os.system('pip install tensorflow')

print('All required libraries are installed.')
