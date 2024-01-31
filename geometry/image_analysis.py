import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.stats
import os
import numpy as np
from PIL import Image
import random
from scipy.stats import norm
from . import decode

# Gray image simulation
def gray_image_simulation():
    image_path = ['picture/grayscale/', 'picture/encoded_image/', 'picture/decoded_image/']
    # 讀取多張圖片
    image_name = [['t001.png', 't002.png', 't003.png', 't004.png'], 
                  ['encoded_t001.png', 'encoded_t002.png', 'encoded_t003.png', 'encoded_t004.png'], 
                  ['decoded_t001.png', 'decoded_t002.png', 'decoded_t003.png', 'decoded_t004.png']]
    image_title = ['Plaintext', 'Encrypted', 'Decrypted']
    # 創建2x4的子圖布局
    fig, axes = plt.subplots(len(image_name[0]),len(image_title), figsize=(12,6))
    fig.suptitle('Gray Analysis', fontsize = 20)
    fword_count = (chr(97+i) for i in range(len(image_name[0])*len(image_title)))
    bword_count = (chr(97+i) for i in range(len(image_name[0])*len(image_title)))
    for i in range(len(image_name[0])):
        for j in range(len(image_title)):
            image = img.imread(os.path.join(image_path[j]+image_name[j][i]))
            ax = axes[i, j]
            ax.imshow(image, cmap = 'gray')
            if j == 0:
                ax.text(0.5,-0.1,f'({next(fword_count)}) {image_title[j]} {image_name[0][i][0:4]}',ha = 'center', va='center', transform=ax.transAxes, fontsize = 10)
                ax.axis('off')
            else:
                ax.text(0.5,-0.1,f'({next(fword_count)}) {image_title[j]} ({next(bword_count)})', ha = 'center', va='center', transform=ax.transAxes, fontsize = 10)
                ax.axis('off')
        next(bword_count)
    plt.tight_layout()
    plt.show()
# key sensitivity
def key_sensitivity_analysis():
    image_path = 'picture/keysense_encode/'
    # 讀取三張灰階圖片
    image1 = plt.imread(os.path.join(image_path+'5C.png'))  
    image2 = plt.imread(os.path.join(image_path+'5D.png'))  
    image3 = plt.imread(os.path.join(image_path+'5E.png'))
    # 比較兩張圖片，生成不同的像素矩陣
    difference1 = np.abs(image1 - image2)
    difference2 = np.abs(image1 - image3)
    compare = [[image1, image2, difference1], [image1, image3, difference2]]
    fig, axes = plt.subplots(2,3, figsize=(12,8))
    fig.suptitle("Key sensitivity analysis", fontsize = 20, y = 0.95)
    title = [['(a) 5C','(b) 5D', '(c) difference between (a) and (b)'], ['(d) 5C','(e) 5E', '(f) difference between (d) and (e)']]
    # 創建一張新的圖片，將不同的像素矩陣轉換為圖片
    for i in range(len(title)):
        for j in range(len(title[i])):
            ax = axes[i,j]
            ax.imshow(compare[i][j], cmap='gray')
            ax.text(0.5,-0.1,title[i][j],size=12,ha="center",transform=ax.transAxes)
            ax.axis('off')
    plt.show()

# Histogram analysis
def histogram_analysis():
    image_path = ['picture/grayscale/', 'picture/encoded_image/']
    def image_2dto1d(image_path):
        image = Image.open(image_path).convert('L')
        image_array = np.array(image).flatten()
        for i in image_array:
            yield i
    # 讀取多張圖片
    image_name = [['t001.png', 't002.png', 't002.png'], ['encoded_t001.png', 'encoded_t002.png', 'encoded_level2_t002.png']]
    image_title = ['Plaintext', 'Encrypted']
    # 創建2x4的子圖布局
    fig, axes = plt.subplots(len(image_name),len(image_name[0]), figsize=(12,6))
    fig.suptitle('Histogram Analysis', fontsize = 20)
    for i in range(len(image_name)):
        for j in range(len(image_name[0])):
            image_value = image_2dto1d(os.path.join(image_path[i]+image_name[i][j]))
            hist, bins = np.histogram(list(image_value), bins = 256, range = (0, 256))
            ax = axes[i, j]
            ax.bar(bins[:-1], hist, width = 1)
            ax.set_title(f'{image_title[i]}{image_name[i][j]}', fontsize = 12)
            ax.set_xlabel('Image gray value', fontsize = 10)
            ax.set_ylabel('Items', fontsize = 10)
            ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
# correlation analysis
def correlation_analysis():
    image_path = ['picture/grayscale/', 'picture/encoded_image/']
    def get_values(image_path): # 產生平行值，垂直值，對角值
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        M,N = 512,512
        num_samples = 5000 # 隨機取5000個樣本，原本是10000，但因為我是pair的關係，所以取5000個
        indices = np.random.randint(low = 0, high = image_array.shape[0]*image_array.shape[1], size = num_samples) # 取得樣本的位置
        # np.where(條件值)
        # 平行
        horizel = image_array.flatten()[indices] # 壓平image_array 再利用 indices 的值，即樣本在image_array的序列位置，得到該位置的值
        shift = np.where((indices+1) % image_array.size == 0, indices-1, indices+1) # 原本是只要將值加1就可以得到右邊的值，即可以有左右平行的值，但因為如果今天抽到最後一個值的話，右邊就沒有值了，所以在這裡使用餘數的概念，如果和image_array長度餘數等於0的話，就取左邊一個值
        horizer = image_array.flatten()[shift]
        # # 垂直
        jump = np.where(np.isin(image_array.flatten()[indices], image_array[-1]), indices-M, indices+N) # 使用 np.isin()的原因是，檢查image_array.flatten()[indices]，是否在 image_array[-1]，即抽到的值是否在最後一行，因為最後一行，沒有往下垂直的值，所以如果是最後一行的話，這裡是往上取值，正常是往下取值
        verticalu = image_array.flatten()[indices]
        verticald = image_array.flatten()[jump]
        # # 對角
        skyline = []
        for index in indices:
            while index % image_array.shape[1] == 0 or index // image_array.shape[1] == image_array.shape[0] - 1:
                index = np.random.randint(low=0, high=image_array.shape[0] * image_array.shape[1])
            skyline.append(index)
        skyline = np.array(skyline)
        diagonalr = image_array.flatten()[skyline]
        diagonall = image_array.flatten()[skyline + M - 1]
        values = [[horizel, horizer], [verticalu, verticald], [diagonalr, diagonall]] # 平行值 ， 垂直值 ， 對角值
        return values
    def generate_correlation_values(values):
        # 計算相關性係數
        correlation_horizontal = np.corrcoef(values[0][0],values[0][1])[0, 1]
        correlation_vertical = np.corrcoef(values[1][0], values[1][1])[0, 1]
        correlation_diagonal = np.corrcoef(values[2][0], values[2][1])[0, 1]
        correlation_values = [correlation_horizontal, correlation_vertical, correlation_diagonal]
        return correlation_values
    # 圖片名稱
    image_names = [['t002.png'], ['encoded_level2_t002.png']]
    image_title = ['Correlation_Horizontal', 'Correlation_Vertical', 'Correlation_Diagonal']
    labels = [['horizel', 'horizer'], ['verticalu', 'verticald'], ['diagonalr', 'diagonal']]
    # 創建子圖
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Correlation Analysis(Test picture:t002.png)', fontsize = 20)
    for i in range(2):
        values = get_values(os.path.join(image_path[i]+image_names[i][0]))
        correlation_values = generate_correlation_values(values)
        for j in range(3):
            ax = axes[i, j]
            ax.scatter(values[j][0], values[j][1], s = 1)
            ax.set_xlabel(labels[j][0])
            ax.set_ylabel(labels[j][1])
            ax.set_title(f'{image_title[j]} :{correlation_values[j]:.2f}')

    # 調整子圖間的間距
    plt.tight_layout()
    plt.show()

# X square Test
def X_square_test():
    alpha = 0.05
    degrees_of_freedom = 255 # 自由度 256個灰值， 自由度為 256 - 1
    critical_value = scipy.stats.chi2.ppf(1 - alpha, degrees_of_freedom) # 臨界值
    # 圖像的大小
    image_size = 512 * 512
    # 每個像素值的預期次數
    expected_counts = image_size / 256
    image_path = ['picture/grayscale/', 'picture/encoded_image/']
    def image_2dto1d(image_path):
        image = Image.open(image_path).convert('L')
        image_array = np.array(image).flatten()
        for i in image_array:
            yield i
    # 讀取多張圖片
    image_name = ['t001.png', 't002.png', 't003.png', 't004.png']
    encoded_image_name = ['encoded_t001.png', 'encoded_t002.png', 'encoded_t003.png', 'encoded_t004.png']
    level2_encoded_image_name = ['encoded_level2_t001.png', 'encoded_level2_t002.png', 'encoded_level2_t003.png','encoded_level2_t004.png']
    # 創建1x4的子圖
    fig, axes = plt.subplots(1,4, figsize=(12,3))
    fig.suptitle('X Square Test', fontsize = 20)
    for i in range(4):
        image_array_P = image_2dto1d(os.path.join(image_path[0]+image_name[i]))
        image_array_C = image_2dto1d(os.path.join(image_path[1]+level2_encoded_image_name[i]))
        # 計算實際的像素值頻率
        observed_counts_P = np.bincount(list(image_array_P), minlength=256)
        observed_counts_C = np.bincount(list(image_array_C), minlength=256)
        # 計算卡方值
        chi_squared_value_P = np.sum((observed_counts_P - expected_counts) ** 2 / expected_counts)
        chi_squared_value_C = np.sum((observed_counts_C - expected_counts) ** 2 / expected_counts)
        if chi_squared_value_C < critical_value:
            PASS = 'Pass'
        else:
            PASS = 'Fail'
        image = img.imread(os.path.join(image_path[0]+image_name[i]))
        ax = axes[i]
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'{image_name[i]}')
        # 在這裡添加顏色設定
        ax.text(0.5, -0.4, f'Plain: {chi_squared_value_P:.2f} \nCipher: {chi_squared_value_C:.2f} \n Pass/Fail: {PASS}', size=12, ha="center", transform=ax.transAxes)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Information entropy analysis
def entropy_analysis():
    # 打開圖像
    image_path = ['picture/grayscale/', 'picture/encoded_image/']
    image_name = ['t001.png', 't002.png', 't003.png', 't004.png']
    encoded_image_name = ['encoded_t001.png', 'encoded_t002.png', 'encoded_t003.png', 'encoded_t004.png']
    level2_encoded_image_name = ['encoded_level2_t001.png', 'encoded_level2_t002.png', 'encoded_level2_t003.png','encoded_level2_t004.png']
    # 創建多張圖片
    fig, axes = plt.subplots(1,4, figsize=(12,3))
    fig.suptitle('Information Entropy Analysis', fontsize = 20)
    for i in range(4):
        image_P = Image.open(os.path.join(image_path[0]+image_name[i])).convert('L')  # 將圖像轉換為灰階模式
        image_C = Image.open(os.path.join(image_path[1]+level2_encoded_image_name[i])).convert('L')  # 將圖像轉換為灰階模式
        # 將圖像轉換為NumPy數組
        image_array_P = np.array(image_P)
        image_array_C = np.array(image_C)
        # 計算像素值的直方圖
        histogram_P = np.histogram(image_array_P, bins=256, range=(0, 256))[0]
        histogram_C = np.histogram(image_array_C, bins=256, range=(0, 256))[0]
        # # 計算概率
        probability_P = histogram_P / np.sum(histogram_P)
        probability_C = histogram_C / np.sum(histogram_C)
        # 計算信息熵
        entropy_P = -np.sum(probability_P * np.log2(probability_P + np.finfo(float).eps)) # np.finfo(float).eps 確保值不會失真，因為log2(N)，如果N越接近0，就會導致，其值不穩定。
        entropy_C = -np.sum(probability_C * np.log2(probability_C + np.finfo(float).eps)) # np.finfo(float).eps 確保值不會失真，因為log2(N)，如果N越接近0，就會導致，其值不穩定。
        ax = axes[i]
        image = img.imread(os.path.join(image_path[0]+image_name[i]))
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'{image_name[i]}')
        ax.text(0.5, -0.4, f'Plain  :{entropy_P:10.3f}\n Cipher : {entropy_C:10.3f}', size=12, ha="center", transform=ax.transAxes)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Robustness analysis(韌性測試)
def robustness_analysis(secret_key):
    # 添加Salt and Pepper噪音
    def s_p_noise(noise_density, original_image, image_name):
        noise_density = noise_density  # 噪音密度，即影響像素的比例
        noise = np.random.choice([0, 255], size=(512, 512), p=[1 - noise_density, noise_density]).astype(np.uint8)
        image = Image.fromarray(np.asarray(original_image)+noise)
        image.save(f"picture/robustness/noisy_{image_name}.png",'png')

    def loss625(original_image, image_name):
        image = np.array(original_image)
        # 創建一個與圖像大小相同的NumPy數組，初始值為255（白色）
        location = [[63,63],[383,63],[63,383],[383,383]]
        for i in location:
            image[i[1]:i[1]+65, i[0]:i[0]+65] = [0]
        # # 將合併後的數組轉換回Pillow圖像對象
        Image.fromarray(image).save(f'picture/robustness/{image_name}_loss_6.25%.png')
    def loss1625(original_image, image_name):
        image = np.array(original_image)
        # 創建一個與圖像大小相同的NumPy數組，初始值為255（白色）
        location = [[63,63],[383,63],[63,383],[383,383]]
        for i in location:
            image[i[1]:i[1]+104, i[0]:i[0]+104] = [0]
        # # 將合併後的數組轉換回Pillow圖像對象
        Image.fromarray(image).save(f'picture/robustness/{image_name}_loss_16.25%.png')
    # original_image = np.full((512,512), 255, dtype=np.uint8) # 產生全白512x512的numpy數組
    # 將 Robustness 要測試的圖片產生出來
    def generate_bobustness_image(image_name, secret_key,set_mode):
        original_image = Image.open(f'picture/encoded_image/encoded_{image_name}.png').convert('L')
        loss1625(original_image,image_name)
        s_p_noise(0.05, original_image, image_name)
        loss625(original_image, image_name)
        decode.decode(secret_key, f"picture/robustness/noisy_{image_name}.png", pic_save= True, save_path= f'picture/robustness/decoded_image/decoded_noisy_{image_name}.png' ,set_level= set_mode)
        decode.decode(secret_key, f"picture/robustness/{image_name}_loss_6.25%.png", pic_save= True, save_path= f'picture/robustness/decoded_image/decoded_{image_name}_loss6.25.png' ,set_level= set_mode)
        decode.decode(secret_key, f"picture/robustness/{image_name}_loss_16.25%.png", pic_save= True, save_path= f'picture/robustness/decoded_image/decoded_{image_name}_loss16.25.png' ,set_level= set_mode)
    # Step 1 => 產生韌性分析需要的圖
    image_name = 'level2_t002' # 設定要做測試的圖片名稱
    set_mode = '2'
    # generate_bobustness_image(image_name, secret_key, set_mode) # 先產生韌性分析需要的圖
    # # Step 2 => 將對應圖表做出來
    image_folder = ['picture/robustness/', 'picture/robustness/decoded_image/']
    original_image_name = [f'noisy_{image_name}.png', f'{image_name}_loss_6.25%.png', f'{image_name}_loss_16.25%.png']
    decoded_image_name = [f'decoded_noisy_{image_name}.png', f'decoded_{image_name}_loss6.25.png', f'decoded_{image_name}_loss6.25.png']
    image_name = [original_image_name, decoded_image_name]
    text_name = [['(a) Salt & pepper noise under 0.05 tensity', '(b) Data loss under 6.25%', '(c) Data loss under 16.25%'],
                 ['(d) Decrypted (a)', '(e) Decrypted(b)', '(f) Decrypted(c)']]
    fig, axes = plt.subplots(2,3, figsize=(12,8))
    fig.suptitle("Robustness Analysis", fontsize = 20, ha= 'center')
    for i in range(2):
        for j in range(3):
            image = img.imread(os.path.join(image_folder[i]+image_name[i][j]))
            ax = axes[i, j]
            ax.imshow(image, cmap='gray')
            ax.text(0.5,-0.05,f'{text_name[i][j]}', ha = 'center', va='center', transform=ax.transAxes, fontsize = 12)
            ax.axis('off')
    plt.tight_layout()
    plt.show()