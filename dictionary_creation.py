import os
import pickle

def load_pickle_files(directory):
    """
    Load all .pk files from the given directory into a dictionary.
    
    :param directory: Directory containing .pk files.
    :return: A dictionary with the first ten characters of filenames as keys and file contents as values.
    """
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pk"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
                key = os.path.splitext(filename)[0][:10]  # Get first ten characters of filename without extension
                data[key] = content
    return data

# Load data from .pk files into a dictionary
directory_path = "/home/Zilan/Desktop/leniasearch/ML_part/Hashed"
loaded_data = load_pickle_files(directory_path)

# Save the dictionary into a new .py file
output_file = "/home/Zilan/Desktop/leniasearch/ML_part/dictionary_data.py"
with open(output_file, 'w') as f:
    f.write(repr(loaded_data))

print("Dictionary saved to {}".format(output_file))

# Assuming loaded_data is your dictionary
size_of_dictionary = len(loaded_data)
print(size_of_dictionary)
#print(loaded_data["9fd8a6a13a"])