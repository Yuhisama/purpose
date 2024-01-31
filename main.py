# from geometry import encode as encode
from geometry import image_analysis as image_analysis
from geometry import encode
from geometry import decode
# image path
set_pic_nums = 4 # 要幾張圖片
image_path = [f'picture/grayscale/t00{i+1}.png' for i in range(set_pic_nums)]
# encode path
encode_save_path = [f'picture/encoded_image/encoded_t00{i+1}.png' for i in range(set_pic_nums)]
level2_encode_save_path = [f'picture/encoded_image/encoded_level2_t00{i+1}.png' for i in range(set_pic_nums)]
# decode path
decode_save_path = [f'picture/decoded_image/decoded_t00{i+1}.png' for i in range(set_pic_nums)]
level2_decode_save_path = [f'picture/decoded_image/decoded_level2_t00{i+1}.png' for i in range(set_pic_nums)]
# # secret key 
member = 4 # 有幾個使用者
secret_key = [i.strip() for i in encode.secret_key(member)] # 4 組金鑰
# Call encode part
# for i in range(set_pic_nums):
#     encode.encode(secret_key[0], image_path[i], pic_save = True, save_path = level2_encode_save_path[i], set_level= '2')
# encode.encode(sk, image_path[1],pic_save = True, save_path = level2_encode_save_path[1],set_level= '2')
# Call decode part
# for i in range(set_pic_nums):
#     decode.decode(secret_key[0], level2_encode_save_path[i], pic_save= True, save_path= level2_decode_save_path[i] ,set_level= '2')
# decode.decode(secret_key[0], level2_encode_save_path[1] ,set_level= '2')
# Simulation and Performance analysis of gray image
# 以下的是加密分析部分
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Gray image simulation
# image_analysis.gray_image_simulation()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Key Sensitivity analysis
# 先產生不同些許不同參數但加密相同圖片
K5C = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C'
K5D = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5D'
K5E = '5C5C5C5C5C5C5C5C5C5C5C5C5C5C5C5E'
image_name = {'5C': K5C, '5D': K5D, '5E': K5E} # original image 是沒有動參數的的加密圖，後兩者都有動到極小的值
# for i in image_name.keys():
#     image_path = f'picture/grayscale/t002.png'
#     encoded_image_name = i
#     key_sensitivity_parameters = image_name[i]
#     keysense_path = 'picture/keysense_encode' # 設定 keysense image 存放資料夾位置
#     encode.encode(key_sensitivity_parameters, image_path,pic_save = True, save_path = f'{keysense_path}/{i}.png',set_level= '2')
# 執行 Key sensitivity analysis
image_analysis.key_sensitivity_analysis()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Histogram analysis 
# image_analysis.histogram_analysis()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Correlation analysis
# image_analysis.correlation_analysis()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# X Square Test
# image_analysis.X_square_test()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Information Entropy analysis
# image_analysis.entropy_analysis()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Robustness analysis
# image_analysis.robustness_analysis(secret_key[0])