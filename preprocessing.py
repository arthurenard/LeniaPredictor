import torch
from dictionary_data import dictionary_data  


dict_data_one_params = {}

for key, value in dictionary_data.items():
    params_a = value['params_a']
    params_d = value['params_d']
    t_crit = value['t_crit']

    new_params = {}
    for param_key in params_a.keys():
        new_params[param_key] = params_d[param_key] * t_crit + params_a[param_key] * (1 - t_crit)

    dict_data_one_params[key] = {'params': new_params}

target_device = 'cuda:0'  

# New dictionary to store the concatenated tensors
dict_data_tensors = {}

# Iterate over the items in the dict_data_one_params
for key, value in dict_data_one_params.items():
    params = value['params']

    concatenated_tensor = torch.cat([
        params[param_key].to(target_device).flatten() for param_key in params.keys()
    ])

    dict_data_tensors[key] = concatenated_tensor

# Path to save the file
file_path = 'dict_data_tensors.pth'

# Save final_dictionary_data to a file
torch.save(dict_data_tensors, file_path)