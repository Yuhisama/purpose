import numpy as np
# secret_key = '1c12f81d63a215e90951ac29afa5c03d'
K5C = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C'
K5D = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5D'
K5E = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5E'
image_name = {'5C': K5D, '5D': K5D, '5E': K5E} 
for i in image_name.keys():
    x_initial = np.float64(int(image_name[i][0:12], 16) / (2**48))  
    r_values = np.float64(int(image_name[i][12:28], 16) / (2**64)) *5 
    pur_item = np.int64(int(image_name[i][28:], 16))
    print(image_name[i][28:])
    ca_rule = np.int8(pur_item % 70)
    modes = np.int8(ca_rule % 9)
    print(f'x_initial -> {x_initial}')
    print(f'r_values -> {r_values}')
    print(f'pur_item -> {pur_item}')
    print(f'ca_rule -> {ca_rule}')
    print(f'modes -> {modes}')
    print('--------------------------------------------------')