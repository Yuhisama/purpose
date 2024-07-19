# from geometry import encode as encode
from geometry import image_analysis as image_analysis
from geometry import encode
from geometry import decode
# standard test images
standard_test_images = ['baboon', 'cameraman', 'house', 'jetplane', 'peppers', 'pirate', 'lake']
# image path
set_pic_nums = len(standard_test_images) # 要幾張圖片
image_path = [f'picture/grayscale/{standard_test_images[i]}.png' for i in range(set_pic_nums)]
# encode path
encode_save_path = [f'picture/encoded_image/encoded_{standard_test_images[i]}.png' for i in range(set_pic_nums)] 
level2_encode_save_path = [f'picture/encoded_image/encoded_level2_{standard_test_images[i]}.png' for i in range(set_pic_nums)]
# decode path
decode_save_path = [f'picture/decoded_image/decoded_{standard_test_images[i]}.png' for i in range(set_pic_nums)]
level2_decode_save_path = [f'picture/decoded_image/decoded_level2_{standard_test_images[i]}.png' for i in range(set_pic_nums)]
# # secret key 
member = len(standard_test_images) # 有幾個使用者
secret_key = [i.strip() for i in encode.secret_key(member)] # 4 組金鑰
# Call encode part
# for i in range(set_pic_nums):
#     encode.encode(secret_key[0], image_path[i], pic_save = True, save_path = level2_encode_save_path[i], set_level= '2')
# encode.encode(secret_key[0], image_path[1],set_level= '2')
# Call decode part
# for i in range(set_pic_nums):
#     decode.decode(image_path[i],secret_key[0], level2_encode_save_path[i], pic_save= True, save_path= level2_decode_save_path[i] ,set_level= '2')
# decode.decode(secret_key[0], level2_encode_save_path[1] ,set_level= '2')
# Simulation and Performance analysis of gray image
# 以下的是加密分析部分
# 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Gray image simulation
# image_analysis.gray_image_simulation(standard_test_images[0:4])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Key Sensitivity analysis
# 先產生不同些許不同參數但加密相同圖片
# K5C = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C'
# K5D = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5D'
# K5E = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5E'
# image_name = {'5C': K5C, '5D': K5D, '5E': K5E} # original image 是沒有動參數的的加密圖，後兩者都有動到極小的值
# for i in image_name.keys():
#     image_path = f'picture/grayscale/baboon.png'
#     encoded_image_name = i
#     key_sensitivity_parameters = image_name[i]
#     keysense_path = 'picture/keysense_encode' # 設定 keysense image 存放資料夾位置
#     encode.encode(key_sensitivity_parameters, image_path,pic_save = True, save_path = f'{keysense_path}/{i}.png',set_level= '2')
# 執行 Key sensitivity analysis
# image_analysis.key_sensitivity_analysis()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Histogram analysis 
# image_analysis.histogram_analysis(standard_test_images[0:4])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Correlation analysis
# image_analysis.correlation_analysis()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# X Square Test
# image_analysis.X_square_test(standard_test_images[0:4])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Information Entropy analysis
image_analysis.entropy_analysis(standard_test_images[0:4])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Local information entropy analysis
# log -> 這裡的結果時常會是Fail，但正常來說 Entropy 越高越好，但論文中統計出來的 Entropy 範圍，就是 [7.901515698, 7.903422936]。
# 這裡使用while True 是因為常常會fail，所以使用while True 直到沒個圖片都pass
# while True:
#     result, PFS , counter =image_analysis.local_entropy_analysis(standard_test_images) # result 為每個圖片的Local Entropy 值，PFS為圖片的PASS，COUNTER為全部圖片PASS數
#     print(result, PFS,counter)
#     if counter == len(standard_test_images):
#         image_analysis.local_entropy_analysis_view(standard_test_images,result, PFS)
#         break
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Robustness analysis
# image_analysis.robustness_analysis(secret_key[0])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Pixel disparity analysis
# image_analysis.pixel_disparity_analysis(standard_test_images)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Differential attack analysis
# image_analysis.differential_attack_analysis(secret_key[0], standard_test_images)
# image_analysis.differential_attack_50_times_analysis(secret_key[0], standard_test_images[0:4])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Resist known-plaintext and chosen-plaintext attacks analysis
# image_analysis.ResistKP_and_ChosenP_Attack_analysis(secret_key[0])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
