from pathlib import Path

# Get the directory containing the current script
current_dir = Path(__file__).parent

# Construct a path relative to the current script
relative_path = current_dir / 'relative' / 'path' / 'to' / 'file.txt'


import os

# Get the current working directory
current_dir = os.getcwd()

# Construct a path relative to the current working directory
relative_path = os.path.join(current_dir, 'relative', 'path', 'to', 'file.ext')

print(relative_path)  # Prints the absolute path to 'file.ext'


# Get the directory containing the current script or module
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct a path relative to the script or module directory
relative_path = os.path.join(script_dir, 'relative', 'path', 'to', 'file.ext')

print(relative_path)  # Prints the absolute path to 'file.ext'


