from PIL import Image
import numpy as np
import os
def image_2dto1d(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image).flatten()
    return image_array.astype('uint8')
def logistic_map(r, x, pur_item):
    return (r * x * (1 - x)*pur_item) % 1
input_image_1d = image_2dto1d('picture/grayscale/lena.png') # 輸入圖片轉成 1D 的 array
catest = image_2dto1d('picture/NPCR_UACI/change_one_pixel_lena.png')

secret_key = '1c12f81d63a215e90951ac29afa5c03d'
x_initial = np.float64(int(secret_key[0:12], 16) / (2**48))   # x 初始值
r_values = np.float64(int(secret_key[12:28], 16) / (2**64))  # 基礎系統參數，進入chaos系統時，三個混沌分別都會乘自己系統參數的範圍最大值
pur_item = np.int64(int(secret_key[28:], 16)) # 提出的擴展參數
# ca_rule = np.int8(pur_item % 70) # CA規則表對應的位置，有做好的balance rule的陣列，這裡就叫CA規則表
# modes = np.int8(ca_rule % 9) # 選擇9個模式的變數

a = np.mean(input_image_1d) % 1
b = np.mean(catest) % 1
a_l = logistic_map(r_values, a, pur_item)
b_l = logistic_map(r_values, b, pur_item)
test = a_l + x_initial
print(x_initial)
print(test)


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