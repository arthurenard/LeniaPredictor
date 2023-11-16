import torch
import json
from model import EnsemblePredictor, SymmetricDNN

dict_data_tensors = torch.load('./dict_data_tensors_test.pth')
#torch.load('./dict_data_tensors.pth') dict_data_tensors_test =

with open('./annotations_data.json', 'r') as file:
    score_data = json.load(file)

inputs_left = []
inputs_right = []
outputs = []

for matchup in score_data:
    left_params = dict_data_tensors.get(matchup['left'])
    right_params = dict_data_tensors.get(matchup['right'])
    
    if left_params is not None and right_params is not None:
        inputs_left.append(left_params)
        inputs_right.append(right_params)
        
        outputs.append(matchup['side'])

# Inputs creation
inputs_left = torch.stack(inputs_left)
inputs_right = torch.stack(inputs_right)
outputs = torch.tensor(outputs, dtype=torch.float32)[:,None] # Make it (B,1) otherwise pytorch complains when computing loss
 

ensemble = EnsemblePredictor(base_model_class=SymmetricDNN, num_predictors=5, 
                             input_dim=2*inputs_left.size(1), hidden_layers=[1000, 6], device="cuda:0")
print("Ensemble input size: ", inputs_left.size(1))

ensemble.train(x=inputs_left, y=inputs_right, true_label=outputs, epochs=100000, batch_size=20)