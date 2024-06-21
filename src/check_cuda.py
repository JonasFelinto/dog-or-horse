
'''
python -m src.check_cuda
'''

import torch
# print(torch.zeros(1).cuda())
print("Cuda available: ", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name())