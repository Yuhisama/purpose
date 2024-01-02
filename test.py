import numpy as np
import time
import secrets
def logistic_map(r, x, pur_item):
    return r * x * (1 - x)*pur_item % 1
def test(iterations, r , x , pur_item):
    for i in range(iterations):
        x = logistic_map(r, x, pur_item)
        yield x

def decimal_to_binary(decimal_num):
    binary_str = bin(decimal_num)[2:]  # 將十進制轉換為二進制，並去掉前綴'0b'
    num_zeros_to_add = 64 - len(binary_str)  # 計算需要補零的位數
    if num_zeros_to_add > 0:
        binary_str = '0' * num_zeros_to_add + binary_str
    return binary_str
random_number = secrets.token_hex(16)
x_initial = np.float64(int(random_number[0:12], 16) / (2**48))  
r_values = np.float64(int(random_number[12:28], 16) / (2**64)) *5
pur_item = np.int64(int(random_number[28:], 16))
ca_rule = np.int8((int(random_number[28:], 16)) % 70)
modes = ca_rule % 9
b = test(100, x_initial, r_values, pur_item)
final = 0
for i in range(100):
    final = next(b)
print(final* (10**20))
final_int = np.uint64(final * (10**20) % (2**64))
fianl_bin = decimal_to_binary(final_int)
print(len(fianl_bin))
print(fianl_bin)