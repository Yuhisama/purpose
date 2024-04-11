import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.stats
import os
import numpy as np
from PIL import Image
import random
from scipy.stats import norm
from . import decode, encode
# Gray image simulation
def gray_image_simulation(image_list):
    image_path = ['picture/grayscale/', 'picture/encoded_image/', 'picture/decoded_image/']
    # 讀取多張圖片
    image_name = [image_list, 
                  [f'encoded_level2_{i}' for i in image_list], 
                  [f'decoded_level2_{i}' for i in image_list]]
    image_title = ['Plaintext', 'Encrypted', 'Decrypted']
    # 創建2x4的子圖布局
    fig, axes = plt.subplots(len(image_name[0]),len(image_title), figsize=(12,6))
    fig.suptitle('Gray Analysis', fontsize = 20)
    fword_count = (chr(97+i) for i in range(len(image_name[0])*len(image_title)))
    bword_count = (chr(97+i) for i in range(len(image_name[0])*len(image_title)))
    for i in range(len(image_name[0])):
        for j in range(len(image_title)):
            image = img.imread(os.path.join(image_path[j]+image_name[j][i]+'.png'))
            ax = axes[i, j]
            ax.imshow(image, cmap = 'gray')
            if j == 0:
                ax.text(0.5,-0.1,f'({next(fword_count)}) {image_title[j]} {image_name[0][i].capitalize()}',ha = 'center', va='center', transform=ax.transAxes, fontsize = 10)
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
def histogram_analysis(image_list):
    image_path = ['picture/grayscale/', 'picture/encoded_image/']
    def image_2dto1d(image_path):
        image = Image.open(image_path).convert('L')
        image_array = np.array(image).flatten()
        for i in image_array:
            yield i
    # 讀取多張圖片
    image_name = [image_list, 
                  [f'encoded_level2_{i}' for i in image_list]]
    image_title = ['Plaintext', 'Encrypted']
    # (a) (b) 標題使用的
    fword_count = (chr(97+i) for i in range(len(image_name[0])*len(image_title)))
    bword_count = (chr(97+i) for i in range(len(image_name[0])*len(image_title)))
    # 創建2x4的子圖布局
    fig, axes = plt.subplots(len(image_name),len(image_name[0]), figsize=(12,6))
    fig.suptitle('Histogram Analysis', fontsize = 20)
    for i in range(len(image_name)):
        for j in range(len(image_name[0])):
            image_value = image_2dto1d(os.path.join(image_path[i]+image_name[i][j]+'.png'))
            hist, bins = np.histogram(list(image_value), bins = 256, range = (0, 256))
            ax = axes[i, j]
            ax.bar(bins[:-1], hist, width = 1)
            if i == 0:
                ax.text(0.5,-0.3,f'({next(fword_count)}){image_title[i]} {image_name[i][j].capitalize()}',size=12,ha="center",transform=ax.transAxes)
            else:
                ax.text(0.5,-0.3,f'({next(fword_count)}){image_title[i]} ({next(bword_count)})',size=12,ha="center",transform=ax.transAxes)
            ax.set_xlabel('Image gray value', fontsize = 10)
            ax.set_ylabel('Items', fontsize = 10)
            ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
# correlation analysis 這邊使用lena
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
    image_names = [['lena'], ['encoded_level2_lena']]
    image_title = ['Correlation_Horizontal', 'Correlation_Vertical', 'Correlation_Diagonal']
    labels = [['horizel', 'horizer'], ['verticalu', 'verticald'], ['diagonalr', 'diagonal']]
    # 創建子圖
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Correlation Analysis(Test picture:lena.png)', fontsize = 20)
    for i in range(2):   
        values = get_values(os.path.join(image_path[i]+image_names[i][0]+'.png'))
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
def X_square_test(image_list, chose_image_level = 2): # chose_image_level 默認是 level2 的加密圖，但可以選level1的
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
    image_name = [image_list,
                  [f'encoded_{i}' for i in image_list],
                  [f'encoded_level2_{i}' for i in image_list]]
    # 創建1x4的子圖
    fig, axes = plt.subplots(1,4, figsize=(12,3))
    fig.suptitle('X Square Test', fontsize = 20)
    for i in range(4):
        image_array_P = image_2dto1d(os.path.join(image_path[0]+image_name[0][i]+'.png'))
        image_array_C = image_2dto1d(os.path.join(image_path[1]+image_name[chose_image_level][i]+'.png'))
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
        image = img.imread(os.path.join(image_path[0]+image_name[0][i]+'.png'))
        ax = axes[i]
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'{image_name[0][i]}')
        # 在這裡添加顏色設定
        ax.text(0.5, -0.4, f'Plain: {chi_squared_value_P:.2f} \nCipher: {chi_squared_value_C:.2f} \n Pass/Fail: {PASS}', size=12, ha="center", transform=ax.transAxes)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Information entropy analysis
def entropy_analysis(image_list, chose_image_level = 2):
    # 打開圖像
    image_path = ['picture/grayscale/', 'picture/encoded_image/']
    image_name = [image_list,
                [f'encoded_{i}' for i in image_list],
                [f'encoded_level2_{i}' for i in image_list]]
    # 創建多張圖片
    fig, axes = plt.subplots(1,4, figsize=(12,3))
    fig.suptitle('Information Entropy Analysis', fontsize = 20)
    for i in range(4):
        image_P = Image.open(os.path.join(image_path[0]+image_name[0][i]+'.png')).convert('L')  # 將圖像轉換為灰階模式
        image_C = Image.open(os.path.join(image_path[1]+image_name[chose_image_level][i]+'.png')).convert('L')  # 將圖像轉換為灰階模式
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
        image = img.imread(os.path.join(image_path[0]+image_name[0][i]+'.png'))
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'{image_name[0][i]}')
        ax.text(0.5, -0.4, f'Plain  :{entropy_P:10.4f}\n Cipher : {entropy_C:10.4f}', size=12, ha="center", transform=ax.transAxes)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
# Local information entropy analysis
def local_entropy_analysis(image_list, chose_image_level = 2):
    # 打開圖像
    image_path = ['picture/encoded_image/', 'picture/grayscale/']
    image_name = [image_list,
                [f'encoded_{i}' for i in image_list],
                [f'encoded_level2_{i}' for i in image_list]]
    # 測試值是否通過
    counter = 0
    test_value = [7.901515698, 7.903422936]
    result = [0]*len(image_list) # 四個結果
    PFS = [0]*len(image_list) # 4個成績
    for i in range(len(image_list)):
        image = Image.open(os.path.join(image_path[0]+image_name[chose_image_level][i]+'.png')).convert('L')  # 將圖像轉換為灰階模式
        # 定義區塊大小
        k = 30  # 您可以調整這個值以改變區塊的大小
        TB = 1936
        sub_blocks_len = int(TB**0.5) # sub_block的邊長
        # 獲取圖像大小
        width, height = image.size

        # 初始化存儲區塊的列表
        blocks = []

        # 生成隨機的不重疊區塊
        for _ in range(k):
            while True:
                # 隨機生成左上角位置的x和y座標，確保不重疊
                x = random.randint(0, width - sub_blocks_len)
                y = random.randint(0, height - sub_blocks_len)
                
                # print(x, y)
                # 檢查新生成的座標是否與已有的區塊重疊
                overlap = False
                for parameters in blocks:
                    existing_x, existing_y = parameters[0]
                    if x < existing_x + sub_blocks_len and x + sub_blocks_len > existing_x and y < existing_y + sub_blocks_len and y + sub_blocks_len > existing_y:
                        overlap = True
                        break
                
                # 如果不重疊，則添加區塊並退出循環
                if not overlap:
                    block = image.crop((x, y, x + sub_blocks_len, y + sub_blocks_len))
                    parameters = [[x, y], block]
                    blocks.append(parameters)
                    break

        # 初始化結果矩陣
        sum = 0
        # 創建一個新的圖像，用原始圖像填充未選取的區塊，並添加差異
        new_image = Image.new('L', (width, height))
        # 計算信息熵
        for parameter in blocks:
            x, y = parameter[0]
            block = parameter[1]
            new_image.paste(block, (x,y))
            hist, _ = np.histogram(block, bins=256, range=(0, 256))  # 計算直方圖
            hist = hist / np.sum(hist)  # 正規化直方圖
            entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))  # 計算信息熵
            sum += entropy  # 將信息熵值存入結果矩陣
        result[i] = sum/k
        PF = 'Pass' if test_value[0] <= result[i] <= test_value[1] else 'Fail' # pass or fail
        if PF == 'Pass':
            counter += 1
            PFS[i] = PF
    return result, PFS, counter
def local_entropy_analysis_view(image_list,result, PFS):
    fig, axes = plt.subplots(1,len(image_list), figsize=(12,3))
    fig.suptitle('Information Entropy Analysis', fontsize = 20)
    image_path = ['picture/encoded_image/', 'picture/grayscale/']
    image_name = [image_list,
                [f'encoded_{i}' for i in image_list],
                [f'encoded_level2_{i}' for i in image_list]]
    for i in range(len(image_list)):
        ax = axes[i]
        image = img.imread(os.path.join(image_path[1]+image_name[0][i]+'.png'))
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'{image_name[0][i]}')
        ax.text(0.5, -0.4, f'Local Entropy:{result[i]:10.4f}\n Pass/Fail:{PFS[i]}', size=12, ha="center", transform=ax.transAxes)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
# Robustness analysis(韌性測試)
def robustness_analysis(secret_key): # secret key 是要產生測試圖使用的
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
    image_name = 'level2_lena' # 設定要做測試的圖片名稱
    set_mode = '2'
    generate_bobustness_image(image_name, secret_key, set_mode) # 先產生韌性分析需要的圖
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

# Pixel Disparity Analysis
def pixel_disparity_analysis(image_list):
    image_folder = ['picture/grayscale/', 'picture/encoded_image/']
    image_name = [image_list,
            [f'encoded_{i}' for i in image_list],
            [f'encoded_level2_{i}' for i in image_list]]
    M, N = 512,512
    fig, axes = plt.subplots(1,len(image_name[0]), figsize = (18,3))
    fig.suptitle("Pixels Disparity Analysis", fontsize = 20)
    for i in range(len(image_name[0])):
        original_image = Image.open(image_folder[0]+ f'{image_name[0][i]}' + '.png')
        encrypted_image = Image.open(image_folder[1]+f'{image_name[2][i]}'+'.png')

        original_array = np.array(original_image, dtype= np.int64)
        encrypted_array = np.array(encrypted_image, dtype = np.int64)
        c = np.abs(original_array - encrypted_array)
        # # # 計算MSE
        mse = np.mean(np.square(original_array - encrypted_array.astype(np.int64)))

        # # # # 計算PSNR
        max_pixel_value = 255  # 8位元圖像的最大像素值
        psnr = 10 * np.log10((max_pixel_value*max_pixel_value / mse))

        # # # 計算MAE
        mae = np.mean(np.abs(original_array - encrypted_array))
        ax = axes[i]
        image = img.imread(os.path.join(image_folder[0]+image_name[0][i]+'.png'))
        ax.imshow(image, cmap = 'gray')
        if i == 0:
            ax.text(-0.4, -0.5, f'MSE           {mse:10.4f} \nPSNR           {psnr:10.4f}\nMAE           {mae:10.4f}', size=12, ha="left", transform=ax.transAxes)
        else:
            ax.text(0.5, -0.5, f'{mse:10.4f}\n{psnr:10.4f}\n{mae:10.4f}', size=12, ha="center", transform=ax.transAxes)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Differential attack analysis
def differential_attack_analysis(secret_key,image_list):
    # Step 1 => 加密同張圖片，但相差一個Pixel，所以需要先改圖片後儲存在NPCR_UACI資料夾中
    # Load the image
    image_folder = ['picture/grayscale/', 'picture/NPCR_UACI/', 'picture/encoded_image/']
    image_name = [image_list,
            [f'encoded_{i}' for i in image_list],
            [f'encoded_level2_{i}' for i in image_list]]
    # for i in range(len(image_name[0])):
    #     image = Image.open(image_folder[0]+image_name[0][i]+'.png')
    #     image_array = np.array(image)
    #     # Choose a random pixel to change
    #     row = random.randint(0, len(image_array))
    #     col = random.randint(0, len(image_array))
    #     new_pixel_value = random.randint(0, 255)
    #     # Modify the pixel value
    #     image_array[row, col] = new_pixel_value
    #     # Save the modified image
    #     save_path = f'picture/NPCR_UACI/change_one_pixel_{image_name[0][i]}.png'
    #     modified_image = Image.fromarray(image_array.astype(np.uint8))
    #     modified_image.save(save_path, 'png')
    # 加密更改一個Pixel的明文圖片，然後儲存到NPCR_UACU folder
    # for i in range(len(image_name[0])):
    #     encode_image_path = f'{image_folder[1]}change_one_pixel_{image_name[0][i]}.png'
    #     encode_image_name = f'picture/NPCR_UACI/encoded_change_one_pixel_c{image_name[0][i]}.png'
    #     encode.encode(secret_key,   encode_image_path, pic_save = True, save_path = encode_image_name, set_level= '2')
    # UACI & NPCR Analysis
    def calculate_npcr(c1, c2):
        c1 = np.array(c1,dtype=(np.uint8))
        c2 = np.array(c2,dtype=(np.uint8))
        total_pixels = len(c1) * len(c1[0])
        different_pixels = 0

        for i in range(len(c1)):
            for j in range(len(c1[0])):
                if c1[i][j] != c2[i][j]:
                    different_pixels += 1

        npc_r = (different_pixels / total_pixels) * 100
        return npc_r
    def calculate_uaci(c1, c2):
        c1 = np.array(c1,dtype=(np.int16))
        c2 = np.array(c2,dtype=(np.int16))
        diff_array = abs(c1-c2).astype(np.int16)
        max_pixel_value = 255  # 最大像素值
        diff = (diff_array)/max_pixel_value
        uaci = np.mean(diff)*100
        return uaci
    fig, axes = plt.subplots(1,len(image_name[0]), figsize = (18,3))
    fig.suptitle("Differential Attack Analysis", fontsize = 20)
    for i in range(len(image_name[0])):
        c1 = Image.open(os.path.join(image_folder[2]+image_name[2][i]+'.png'))
        c2 = Image.open(os.path.join(image_folder[1]+f'encoded_change_one_pixel_c{image_name[0][i]}.png'))
        npcr = calculate_npcr(c1,c2)
        uaci = calculate_uaci(c1,c2)
        image = img.imread(os.path.join(image_folder[0]+image_name[0][i]+'.png'))
        ax = axes[i]
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'{image_name[0][i]}')
        if i == 0:
                ax.text(-0.5, -0.3, f'NPCR       {npcr:10.4f} \nUACI       {uaci:10.4f}', size=12, ha="left", transform=ax.transAxes)
        else:
            ax.text(0.4, -0.5, f'{npcr:10.4f}\n{uaci:10.4f}\n', size=12, ha="center", transform=ax.transAxes)
        ax.axis('off')
    # plt.tight_layout()
    plt.show()
# 50次的 Differential attack analysis
def differential_attack_50_times_analysis(secret_key,image_list):
    # 此為 50 times 測試的 differential attack analysis
    # Step 1 => 加密同張圖片，但相差一個Pixel，所以需要先改圖片後儲存在NPCR_UACI_ALPHA資料夾中
    # Load the image
    image_folder = ['picture/grayscale/', 'picture/NPCR_UACI_ALPHA/', 'picture/encoded_image/'] # 第一個放[原始圖片資料夾, 更換一個Pixel圖片的資料夾, 原始圖片加密後的圖片]
    image_name = image_list
    # for i in range(len(image_name)):
    #     image = Image.open(image_folder[0]+image_name[i]+'.png')
    #     image_array = np.array(image)
    #     for j in range(50): # 重複 50 次
    #         # Choose a random pixel to change
    #         row = random.randint(0, len(image_array)-1)
    #         col = random.randint(0, len(image_array)-1)
    #         new_pixel_value = random.randint(0, 255)
    #         # Modify the pixel value
    #         image_array[row, col] = new_pixel_value
    #         # Save the modified image
    #         save_path = f'picture/NPCR_UACI_ALPHA/change_one_pixel_{image_name[i]}{j}.png'
    #         modified_image = Image.fromarray(image_array.astype(np.uint8))
    #         modified_image.save(save_path, 'png')
    # Step2 => 加密更改一個Pixel的明文圖片，然後儲存到NPCR_UACU_ALPHA folder
    # for i in range(len(image_name)):
    #     for j in range(50):
    #         encode_image_path = f'{image_folder[1]}change_one_pixel_{image_name[i]}{j}.png' # 欲加密圖片路徑名稱
    #         encode_image_name = f'encode_change_one_pixel_c{image_name[i]}{j}.png' # 加密後的圖片路徑名稱
    #         encode.encode(secret_key,   encode_image_path, pic_save = True, save_path = encode_image_name, set_level= '2')
    #         os.remove(encode_image_path) # 刪除更改一個像素後的明文圖片
    # Step1 和 Step2 第一次執行後就可以 block掉了，因為只有第一次需要產生圖片，產生後就不用了
    # UACI & NPCR Analysis
    def calculate_npcr(c1, c2): # 計算 npcr
        c1 = np.array(c1,dtype=(np.uint8))
        c2 = np.array(c2,dtype=(np.uint8))
        total_pixels = len(c1) * len(c1[0])
        different_pixels = 0

        for i in range(len(c1)):
            for j in range(len(c1[0])):
                if c1[i][j] != c2[i][j]:
                    different_pixels += 1

        npc_r = (different_pixels / total_pixels) * 100
        return npc_r
    def calculate_uaci(c1, c2): # 計算 uaci
        c1 = np.array(c1,dtype=(np.int16))
        c2 = np.array(c2,dtype=(np.int16))
        diff_array = abs(c1-c2).astype(np.int16)
        max_pixel_value = 255  # 最大像素值
        diff = (diff_array)/max_pixel_value
        uaci = np.mean(diff)*100
        return uaci
    def calculate_critical_value(): # 計算臨界值(alpha = 0.05)
        # 給定參數
        F = 255
        alpha = 0.05
        M = 512  # 圖像寬度
        N = 512  # 圖像高度

        # 計算逆標準正態分佈的值
        # z = norm.ppf(alpha)
        f = norm.ppf(alpha/2)
        # 計算結果
        # critical_value = ((F - (z * ((F / (M * N))**0.5))) / (F + 1)) * 100 # 由於論文中他算出來的值是 99.5893% ，但我這已算出來都是 99.60% ，所以先以論文為基準。
        Lower_critical_value = (((F+2)/(3*F+3)) - (f*(((F+2)*((F**2) + (2*F) + 3)/(18 * ((F+1)*(F+1)) * M * N * F)) ** 0.5))) * 100
        Upper_critical_value = (((F+2)/(3*F+3)) + (f*(((F+2)*((F**2) + (2*F) + 3)/(18 * ((F+1)*(F+1)) * M * N * F)) ** 0.5))) * 100
        return [99.5893, Lower_critical_value, Upper_critical_value]

    def get_npcr_uaci_array():
        npcr_2_value, npcr_3_value, npcr_4_value, npcr_5_value = [], [], [], []
        uaci_2_value, uaci_3_value, uaci_4_value, uaci_5_value = [], [], [], []
        value = [[npcr_2_value, npcr_3_value, npcr_4_value, npcr_5_value], 
                [uaci_2_value, uaci_3_value, uaci_4_value, uaci_5_value]]
        for i in range(4):
            for j in range(50):
                c1 = Image.open(os.path.join(image_folder[2]+'encoded_level2_'+image_name[i]+'.png')) # 沒改變的原加密圖
                c2 = Image.open(os.path.join(image_folder[1]+f'encode_change_one_pixel_c{image_name[i]}{j}.png')) # 改變一個pixel的加密圖
                value[0][i].append(calculate_npcr(c1, c2))
                value[1][i].append(calculate_uaci(c1, c2))
        return value
            
    fig , axes = plt.subplots(1,2, figsize = (12,6))
    fig.suptitle("Differential Attack(test 50 times analysis)", fontsize = 20)
    critical_value = calculate_critical_value() # value 分別是 NPCR, UACI_L, UACI_U
    N_U_value = get_npcr_uaci_array() # 值分別是value = [[npcr_2_value, npcr_3_value, npcr_4_value, npcr_5_value], 
                                #                 [uaci_2_value, uaci_3_value, uaci_4_value, uaci_5_value]]
    for i in range(2):
        if i == 0:
            value, value1, value2, value3 = N_U_value[i][0], N_U_value[i][1], N_U_value[i][2], N_U_value[i][3]
            ax = axes[i]
            ax.plot(np.arange(50), value, 'm-', label = f'{image_name[0]}(Mean : {np.mean(value):.3f}%)')
            ax.plot(np.arange(50), value1,'g-', label = f'{image_name[1]}(Mean : {np.mean(value1):.3f}%)')
            ax.plot(np.arange(50), value2, 'b-', label = f'{image_name[2]}(Mean : {np.mean(value2):.3f}%)')
            ax.plot(np.arange(50), value3, 'c-', label = f'{image_name[3]}(Mean : {np.mean(value3):.3f}%)')
            ax.set_ylabel("NPCR%")
            ax.set_ylim(99.54,99.7) # 設定 y軸邊界
            ax.axhline(y=critical_value[0], color='red', linestyle='--', label=f'critical value({critical_value[0]:.3f})') # 設定臨界值
            ax.legend()
        else:
            value, value1, value2, value3 = N_U_value[i][0], N_U_value[i][1], N_U_value[i][2], N_U_value[i][3]
            ax = axes[i]
            ax.plot(np.arange(50), value, 'm-', label = f'{image_name[0]}(Mean : {np.mean(value):.3f}%)')
            ax.plot(np.arange(50), value1,'g-', label = f'{image_name[1]}(Mean : {np.mean(value1):.3f}%)')
            ax.plot(np.arange(50), value2, 'b-', label = f'{image_name[2]}(Mean : {np.mean(value2):.3f}%)')
            ax.plot(np.arange(50), value3, 'c-', label = f'{image_name[3]}(Mean : {np.mean(value3):.3f}%)')
            ax.set_ylabel("UACI%")
            ax.set_ylim(33.3,33.7)
            ax.axhline(y=critical_value[1], color='red', linestyle='--', label=f'Lower_critical value({critical_value[1]:.3f}%)')
            ax.axhline(y=critical_value[2], color='red', linestyle='--', label=f'Upper_critical value({critical_value[2]:.3f}%)')
            ax.legend()
    plt.tight_layout()
    plt.show()

# Resist known-plaintext and chosen-plaintext attacks analysis
def ResistKP_and_ChosenP_Attack_analysis(secret_key):
    RKPCPA_folder ='picture/ResistKP_and_CPA_analysis/'
    image_name = [['white_image.png', 'white_c_image.png','white_c_histogram.png'], # 白圖
                ['black_image.png', 'black_c_image.png','black_c_histogram.png']] # 黑圖
    def image_2dto1d(image_path):
        image = Image.open(image_path).convert('L')
        image_array = np.array(image).flatten()
        for i in image_array:
            yield i
    def histogram_image(image_name):
        for i in range(2):
            image_value = image_2dto1d(os.path.join(RKPCPA_folder,image_name[i][1]))
            hist, bins = np.histogram(list(image_value), bins = 256, range = (0, 256))
            plt.figure(figsize=(8, 6))
            plt.bar(bins[:-1], hist, width=1)
            plt.savefig(os.path.join(RKPCPA_folder, image_name[i][2]))

    # Step 1 => 製作 黑白 圖像並儲存進資料夾中
    # white_array = np.full((512,512), 255, dtype = np.uint8)
    # black_array = np.full((512,512), 0, dtype = np.uint8)
    # white_image = Image.fromarray(white_array)
    # black_image = Image.fromarray(black_array)
    # white_image.save(os.path.join(RKPCPA_folder+'white_image.png'), 'png')
    # black_image.save(os.path.join(RKPCPA_folder+'black_image.png'), 'png')
    # Step 2 => 加密 黑白兩張圖片，將加密圖儲存起來，順便也將histogram圖也存起來
    # encode.encode(secret_key,   os.path.join(RKPCPA_folder+'white_image.png'), pic_save = True, save_path = os.path.join(RKPCPA_folder+'white_c_image.png'), set_level= '2')
    # encode.encode(secret_key,   os.path.join(RKPCPA_folder+'black_image.png'), pic_save = True, save_path = os.path.join(RKPCPA_folder+'black_c_image.png'), set_level= '2')
    # histogram_image(image_name) # 產生直方圖
    # Step 3 => 使用直方圖分析，並將對比圖做出來 
    fig, axes = plt.subplots(2,3, figsize = (18,6))
    fig.suptitle("Resist Known-plaintext and chosen-plaintext attack analysis")
    for i in range(2):
        for j in range(3):
            image = Image.open(os.path.join(RKPCPA_folder,image_name[i][j])).convert('RGB')
            ax = axes[i,j]
            ax.imshow(image)
            ax.set_title(f'{image_name[i][j]}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

