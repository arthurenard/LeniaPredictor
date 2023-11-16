import os
import pickle
import torch

def custom_tensor_repr(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor_repr = repr(tensor).replace('tensor(', 'torch.tensor(')
        return tensor_repr.replace(', device=\'cuda:0\')', ')')
    return repr(tensor)

def format_dict_for_py(data_dict):
    formatted_dict = {}
    for key, value in data_dict.items():
        formatted_value = {}
        for sub_key, sub_value in value.items():
            if sub_key != "k_size":  # Skip "k_size" in nested dictionaries
                formatted_value[sub_key] = custom_tensor_repr(sub_value)
        formatted_dict[key] = formatted_value
    return formatted_dict

def update_and_process_data(input_directory, output_py_file, output_pth_file, target_device='cuda:0'):
    """
        <description>

        Args :
        input_directory : directory containing data
    """
    def load_pickle_files(directory, existing_data):
        for filename in os.listdir(directory):
            if filename.endswith(".pk"):
                key = os.path.splitext(filename)[0][:10]
                # Only add new data
                if key not in existing_data:
                    file_path = os.path.join(directory, filename)
                    with open(file_path, 'rb') as f:
                        content = pickle.load(f)
                        existing_data[key] = content
        return existing_data

    # Load existing data if it exists
    existing_data = {}
    if os.path.exists(output_py_file):
        with open(output_py_file, 'r') as f:
            exec(f.read())
            existing_data = dictionary_data

    # Update existing data with new data from the input directory
    updated_data = load_pickle_files(input_directory, existing_data)

    dict_data_one_params = {}
    for key, value in updated_data.items():
        if "params_a" in value and "params_d" in value and "t_crit" in value:
            params_a = value['params_a']
            params_d = value['params_d']
            t_crit = value['t_crit']

            new_params = {}
            for param_key in params_a.keys():
                if param_key != "k_size":  # Skip "k_size"
                    new_params[param_key] = params_d[param_key] * t_crit + params_a[param_key] * (1 - t_crit)

            dict_data_one_params[key] = {'params': new_params}

    # Format the dictionary for Python file
    formatted_data = format_dict_for_py(dict_data_one_params)

    # Save the formatted dictionary to a .py file
    with open(output_py_file, 'w') as f:
        f.write("import torch\n\n")
        f.write("dictionary_data = ")
        f.write(repr(formatted_data))

    print(f"Updated dictionary saved to {output_py_file}")
    print(f"Size of updated dictionary: {len(formatted_data)}")

    dict_data_tensors = {}
    for key, value in dict_data_one_params.items():
        params = value['params']

        tensors_list = []
        for param_key, param_value in params.items():
            # Ensure param_value is a tensor
            if not isinstance(param_value, torch.Tensor):
                param_value = torch.tensor([param_value], device=target_device)
            else:
                param_value = param_value.to(target_device)
            tensors_list.append(param_value.flatten())

        concatenated_tensor = torch.cat(tensors_list)
        dict_data_tensors[key] = concatenated_tensor

    print(dict_data_tensors)
    torch.save(dict_data_tensors, output_pth_file)
    print(f"Processed data saved to {output_pth_file}")

# Set up
input_directory = "/home/Zilan/Desktop/leniasearch/ML_part/Hashed"
output_py_file = "/home/Zilan/Desktop/leniasearch/ML_part/dictionary_data_test.py"
output_pth_file = "/home/Zilan/Desktop/leniasearch/ML_part/dict_data_tensors_test.pth"

update_and_process_data(input_directory, output_py_file, output_pth_file)
