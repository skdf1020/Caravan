# def recep_cal(kernel, layer):
#     result = 1
#     for i in range(layer):
#         result += (kernel-1)*(2**i+1)
#     return result
#
# print(recep_cal(3, 8))

import numpy as np
import torch
import torch.nn

a = np.array([[0,1],
              [0,1]])
b = np.array([[1, 1],
              [0, 0]])
print(a *b)

# print(np.sum(np.array([[0.6782, -0.9454,  0.7964],
#                        [-0.4481,  1.2796, -0.3829],
#                        [1.0135, -0.1141,  0.9145],
#                        [-0.6902, -0.5975, -0.7088]]) *
#              np.array([[0.1114, -0.2197, -0.0955],
#                        [0.3702,  0.3041, -0.1755],
#                        [0.2317, -0.1827, -0.1392],
#                        [0.2496, -0.1033, -0.4330]])) + 0.0130)

print(np.sum(np.array([[-0.2927, -1.0048, -0.4284]]) *
             np.array([[-0.1015,  0.3382, -0.4336]])) + 0.5383)
#
# a = torch.tensor([[-0.4315, -0.2637, -0.15], [0.4405, -1.3394, 2.0857]])
# b = torch.tensor()