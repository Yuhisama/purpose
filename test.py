from PIL import Image
import numpy as np
import os
import math
import time

def image_2dto1d(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image).flatten()
    return image_array.astype('uint8')
def logistic_map(r, x, pur_item):
    return (r * x * (1 - x)*pur_item) % 1
def tent_map(x, r, pur_item):
    if x < 0.5:
        return (r * x * pur_item) % 1
    else:
        return (r * (1 - x) * pur_item) % 1
def sin_map(x, r, pur_item):
    return (r * math.sin(math.pi * x) * pur_item) % 1
def ca_rule(rules):
    patterns = {} # rules 的序列
    pattern_list = ['000', '001', '010', '011', '100', '101', '110', '111']
    for i in range(7, -1 ,-1):
        if int(rules/(2**i)) == 1:
            patterns[pattern_list[i]] = '1'
            rules = rules - 2**i
        else:
            patterns[pattern_list[i]] = '0'
    return patterns
def random_ca_rule():
    random_rules = []
    for rule in range(256):
        counter = 0
        rules_patterns = ca_rule(rule)
        for value in rules_patterns.values():
            counter += int(value)
        if counter == 4:
            random_rules.append(rule)
    return random_rules
# random rules ()
random_rules = random_ca_rule()
print(random_rules)
# secret_key ='76d7ca04a2feb3739fa840ef591e1006b532f7a348ed58994a7668a8b2b9ab5c'
# input_image_1d = image_2dto1d('picture/grayscale/baboon.png') # 輸入圖片轉成 1D 的 array
# catest = image_2dto1d('picture/NPCR_UACI/change_one_pixel_lena.png')
# mean_input_image_1d = np.mean(input_image_1d) % 1 # 只取小數拿來當special source 的值


# # 原始系統參數
# logistic_r_values = np.float64(int(secret_key[36:44], 16) / (2**32))  # logistic 基礎系統參數 32 bits
# tent_r_values = np.float64(int(secret_key[44:52], 16) / (2**32))  # tent 基礎系統參數 32 bits
# sine_r_values = np.float64(int(secret_key[52:60], 16) / (2**32))  # sine 基礎系統參數 32 bits
# # 提出的擴展系統參數
# pur_item = np.int64(int(secret_key[60:], 16)) # 提出的擴展參數 16 bits
# # print(f'logistic_r_values = {logistic_r_values}, tent_r_values = {tent_r_values}, sine_r_values = {sine_r_values}')
# # print(f'pur_item = {pur_item}')
# # 選取系統要使用哪個 CA rule、選取系統要用哪一個模式組合
# ca_rule = np.int64(int(secret_key[62:], 16) % 70)  # CA規則表對應的位置，有做好的balance rule的陣列，這裡就叫CA規則表
# modes = np.int8(ca_rule % 9) # 選擇9個模式的變數
# # print(f'ca_rule = {ca_rule}')
# # print(f'modes = {modes}')


# special_source = logistic_map(logistic_r_values, mean_input_image_1d, pur_item)
# # 初始值
# logistic_initial = (np.float64(int(secret_key[0:12], 16) / (2**48)) + special_source) % 1  # logistic 初始值 48 bits
# tent_inital = (np.float64(int(secret_key[12:24], 16) / (2**48)) + special_source) % 1 # tent 初始值 48 bits
# sine_inital = (np.float64(int(secret_key[24:36], 16) / (2**48)) + special_source) % 1# sine 初始值 48 bits
# # 初始值和遠始參數組成群組
# initial_values = [logistic_initial, tent_inital, sine_inital] # 將初始值組成群組好管理
# r_values = [logistic_r_values, tent_r_values, sine_r_values]

# def generation_seq(iterations, r_values , initial_values , pur_item, mode):
#     logistic_value , tent_value, sin_value = initial_values[0], initial_values[1], initial_values[2]
#     if mode == '0':
#         for _ in range(iterations):
#             logistic_value = logistic_map(r_values[0], logistic_value, pur_item)
#             tent_value = tent_map(r_values[1], tent_value, pur_item)
#             sin_value = sin_map(r_values[2],sin_value, pur_item)
#             yield logistic_value, tent_value, sin_value
#     elif mode == 'L':
#         for _ in range(iterations):
#             logistic_value = logistic_map(r_values[0], logistic_value, pur_item)
#             yield logistic_value
#     elif mode == 'T':
#         for _ in range(iterations):
#             tent_value = tent_map(r_values[1], tent_value, pur_item)
#             yield tent_value
#     elif mode == 'S':
#         for _ in range(iterations):
#             sin_value = sin_map(r_values[2],sin_value, pur_item)
#             yield sin_value

# seq = generation_seq(100, r_values, initial_values, pur_item, mode = '0') 
# for _ in range(100):
#     logistic_value ,tent_value ,sin_value = next(seq)
# print(logistic_value, tent_value, sin_value)






# a = np.mean(input_image_1d) % 1
# b = np.mean(catest) % 1
# a_l = logistic_map(r_values, a, pur_item)
# b_l = logistic_map(r_values, b, pur_item)
# test = a_l + x_initial
# print(x_initial)
# print(test)


# standard_test_images = ['baboon', 'cameraman', 'lena', 'house', 'jetplane', 'peppers', 'pirate', 'lake']
# image_folder = ['picture/grayscale/', 'picture/NPCR_UACI/','picture/encoded_image/']
# image_name = [standard_test_images,
#         [f'encoded_{i}' for i in standard_test_images],
#         [f'encoded_level2_{i}' for i in standard_test_images]]
# def calculate_npcr(c1, c2):
#     c1 = np.array(c1,dtype=(np.uint8))
#     c2 = np.array(c2,dtype=(np.uint8))
#     total_pixels = len(c1) * len(c1[0])
#     different_pixels = 0

#     for i in range(len(c1)):
#         for j in range(len(c1[0])):
#             if c1[i][j] != c2[i][j]:
#                 different_pixels += 1

#     npc_r = (different_pixels / total_pixels) * 100
#     return npc_r
# def calculate_uaci(c1, c2):
#     c1 = np.array(c1,dtype=(np.int16))
#     c2 = np.array(c2,dtype=(np.int16))
#     diff_array = abs(c1-c2).astype(np.int16)
#     max_pixel_value = 255  # 最大像素值
#     diff = (diff_array)/max_pixel_value
#     uaci = np.mean(diff)*100
#     return uaci
# c1 = Image.open(os.path.join(image_folder[2]+'encoded_level2_'+image_name[0][0]+'.png'))
# c2 = Image.open(os.path.join(image_folder[1]+f'encoded_change_one_pixel_c{image_name[0][0]}.png'))
# c1_arr = np.array(c1)
# c2_arr = np.array(c2)
# print(c1_arr)
# print(c2_arr)
# test = calculate_npcr(c1,c2)
# print(test)
# for i in range(len(standard_test_images)):
#     print(f'i = {i}')
#     print(f'name = {image_folder[0]+ f'{image_name[0][i]}' + '.png'}')
#     original_image = Image.open(image_folder[0]+ f'{image_name[0][i]}' + '.png')
#     encrypted_image = Image.open(image_folder[1]+f'{image_name[2][i]}'+'.png')
#     original_array = np.array(original_image)
#     encrypted_array = np.array(encrypted_image)
#     print(f'original_shape: {original_array.shape}')
#     print(f'original_image:{original_array}')
#     print(f'encrypted_shape:{encrypted_array.shape}')
#     print(f'encrypted_image:{encrypted_array}')
#     print('============================================================')