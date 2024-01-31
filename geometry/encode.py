import numpy as np
from PIL import Image
import math
import time
import os
M,N = 512,512
# tool functions
def image_2dto1d(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image).flatten()
    return image_array.astype('uint8')
def show_image(image_array, pic_save = False, save_path = None):
    test = Image.fromarray(image_array)
    test.show()
    if pic_save:
        test.save(save_path)
def nist_test(seq): # 這會做一個 test.txt 檔，可以拿去測試 NIST TEST。 參數就是產生出來的序列。
    seq_256 = ((seq*(10**17)) % 256).astype('uint8')
    test = (decimal_to_binary(i, 8) for i in seq_256)
    file_name = '../test.txt'
    with open(file_name,'a') as text_file:
        try:
            while True:
                value = [next(test) for _ in range(3)]
                value = ''.join(filter(str.isdigit, value))
                text_file.write(f'{value}\n')
        except StopIteration:
            pass
            print(f'Succes to write in {file_name}')
def change_array_block8x8(array, block_length): # array : 欲拆的數組 ; block_length : 欲拆成的數組邊長
    M,N = array.shape
    subblocks = [
    array[i:i+block_length, j:j+block_length]
    for i in range(0, M, block_length)
    for j in range(0, N, block_length)
    ]
    return subblocks
def combine_array_block8x8(subblocks, subblocks_nums): # subblocks : 子數組 ; subblocks_nums : 子數組的數量
    lens = int(subblocks_nums **(0.5))
    arr_combined =  np.block([[subblocks[row * lens + col] for col in range(lens)] for row in range(lens)])
    return arr_combined
def sort_list(deal_list): # sort tools output => (value, 原始陣列的位置)
    # dl_sort = sorted(deal_list) # dl_sort is deal_list_sort
    # for i in range(len(deal_list)):
    #     return deal_list.index(dl_sort[i])
    indices = np.argsort(deal_list)
    return indices
def block_chages(blocks, s):
    blocks_copy = blocks.copy()
    for i in range(len(blocks)):
        blocks_copy[i] = blocks[s[i]]
    return blocks_copy
# map function
def logistic_map(r, x, pur_item):
    return (r * x * (1 - x)*pur_item) % 1
def tent_map(x, r, pur_item):
    if x < 0.5:
        return (r * x * pur_item) % 1
    else:
        return (r * (1 - x) * pur_item) % 1
def sin_map(x, r, pur_item):
    return (r * math.sin(math.pi * x) * pur_item) % 1
# CA function
# 這兩個是產生 RANDOM_rules 的函式也可以不用
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
# CA main function
def decimal_to_binary(decimal_num, control_bits):
    binary_str = bin(decimal_num)[2:]  # 將十進制轉換為二進制，並去掉前綴'0b'
    num_zeros_to_add = control_bits - len(binary_str)  # 計算需要補零的位數
    if num_zeros_to_add > 0:
        binary_str = '0' * num_zeros_to_add + binary_str
    return binary_str
def cell_automata(string , rules, generations = 1):
    # print(f"Init_string : {string}, rule = {rules}")
    patterns = {} # rules 的序列
    pattern_list = ['000', '001', '010', '011', '100', '101', '110', '111']
    n = len(string)
    for i in range(7, -1 ,-1):
        if int(rules/(2**i)) == 1:
            patterns[pattern_list[i]] = '1'
            rules = rules - 2**i
        else:
            patterns[pattern_list[i]] = '0'
    for j in range(0, generations):
        new_string = ''
        for i in range(0, n):
            pattern = string[i-1] + string[i] + string[(i+1)%n]
            new_string = new_string + patterns[pattern]
        string = new_string 
    return new_string
# test yield
def generation_seq(iterations, r , x_inital , pur_item, mode):
    logistic_value , tent_value, sin_value = x_inital, x_inital, x_inital
    if mode == '0':
        for _ in range(iterations):
            logistic_value = logistic_map(r, logistic_value, pur_item)
            tent_value = tent_map(r, tent_value, pur_item)
            sin_value = sin_map(r,sin_value, pur_item)
            yield logistic_value, tent_value, sin_value
    elif mode == 'L':
        for _ in range(iterations):
            logistic_value = logistic_map(r, logistic_value, pur_item)
            yield logistic_value
    elif mode == 'T':
        for _ in range(iterations):
            tent_value = tent_map(r, tent_value, pur_item)
            yield tent_value
    elif mode == 'S':
        for _ in range(iterations):
            sin_value = sin_map(r,sin_value, pur_item)
            yield sin_value
# switch table
def switch_table(iterations, modes, r_values, x_initial, pur_item, ca_rule):
    # 控制流量
    control_bits = 64
    iterations = iterations
    # Create a list to store the seq values and final values
    final_result = np.empty(iterations)
    # Iterate through different parameter values
    r = r_values
    pur_item = pur_item
    # Iterate to reach a stable state
    seq = generation_seq(100, r, x_initial, pur_item, mode = '0') 
    for _ in range(100):
        logistic_value ,tent_value ,sin_value = next(seq)
    # selec mode
    mode = modes % 9
    logistic_seq = generation_seq(iterations, r , logistic_value, pur_item, mode = 'L')
    tent_seq = generation_seq(iterations,r,tent_value, pur_item, mode = 'T')
    sin_seq = generation_seq(iterations,r, sin_value, pur_item, mode = 'S')
    # main function
    match mode:
        case 0: # Control :L, Seq1:T, Seq2:S
            print("mode 0 => Control:L, Seq1:T, Seq2:S")
            counter = 0
            for _ in range(int(iterations/control_bits)):
                trans_binary = decimal_to_binary(np.uint64(next(logistic_seq) * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        final_result[counter] = next(tent_seq)
                        counter +=1
                    elif switch == '1':
                        final_result[counter] = next(sin_seq)
                        counter +=1
        case 1:
            print("mode 1 => Control:T, Seq1:L, Seq2:S")
            counter = 0
            for _ in range(int(iterations/control_bits)):
                trans_binary = decimal_to_binary(np.uint64(next(tent_seq) * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        final_result[counter] = next(logistic_seq)
                        counter +=1
                    elif switch == '1':
                        final_result[counter] = next(sin_seq)
                        counter +=1
        case 2:
            print("mode 2 => Cotrol:S, Seq1:T, Seq2:L")
            counter = 0
            for _ in range(int(iterations/control_bits)):
                trans_binary = decimal_to_binary(np.uint64(next(sin_seq) * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        final_result[counter] = next(tent_seq)
                        counter +=1
                    elif switch == '1':
                        final_result[counter] = next(logistic_seq)
                        counter +=1
        case 3:
            print("mode 3 => Control:S, Seq1:S, Seq2:L")
            counter = 0
            register_counters = 0 
            for _ in range(int(iterations/control_bits)):
                register = next(sin_seq)
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        if register_counters == 0:
                            final_result[counter] = register
                            register_counters += 1
                            counter+=1
                        else:
                            final_result[counter] = next(sin_seq)
                            counter +=1
                    elif switch == '1':
                        final_result[counter] = next(logistic_seq)
                        counter +=1
                register_counters = 0
        case 4:
            print("mode 4 => Control:S, Seq1:S, Seq2:T")
            counter = 0
            register_counters = 0 
            for _ in range(int(iterations/control_bits)):
                register = next(sin_seq)
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        if register_counters == 0:
                            final_result[counter] = register
                            register_counters += 1
                            counter+=1
                        else:
                            final_result[counter] = next(sin_seq)
                            counter +=1
                    elif switch == '1':
                        final_result[counter] = next(tent_seq)
                        counter +=1
                register_counters = 0
        case 5:
            print("mode 5 => Control:L, Seq1:L, Seq2:S")
            counter = 0
            register_counters = 0 
            for _ in range(int(iterations/control_bits)):
                register = next(logistic_seq)
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        if register_counters == 0:
                            final_result[counter] = register
                            register_counters += 1
                            counter+=1
                        else:
                            final_result[counter] = next(logistic_seq)
                            counter +=1
                    elif switch == '1':
                        final_result[counter] = next(sin_seq)
                        counter +=1
                register_counters = 0
        case 6:
            print("mode 6 => Control:L, Seq1:L, Seq2:T")
            counter = 0
            register_counters = 0 
            for _ in range(int(iterations/control_bits)):
                register = next(logistic_seq)
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        if register_counters == 0:
                            final_result[counter] = register
                            register_counters += 1
                            counter+=1
                        else:
                            final_result[counter] = next(logistic_seq)
                            counter +=1
                    elif switch == '1':
                        final_result[counter] = next(tent_seq)
                        counter +=1
                register_counters = 0
        case 7:
            print("mode 7 => Control:T, Seq1:T, Seq2:L")
            counter = 0
            register_counters = 0 
            for _ in range(int(iterations/control_bits)):
                register = next(tent_seq)
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        if register_counters == 0:
                            final_result[counter] = register
                            register_counters += 1
                            counter+=1
                        else:
                            final_result[counter] = next(tent_seq)
                            counter +=1
                    elif switch == '1':
                        final_result[counter] = next(logistic_seq)
                        counter +=1
                register_counters = 0
        case 8:
            print("mode 8 => Control:T, Seq1:T, Seq2:S")
            counter = 0
            register_counters = 0 
            for _ in range(int(iterations/control_bits)):
                register = next(tent_seq)
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)), control_bits)
                switch_rules = cell_automata(trans_binary, ca_rule)
                for switch in switch_rules:
                    if switch == '0':
                        if register_counters == 0:
                            final_result[counter] = register
                            register_counters += 1
                            counter+=1
                        else:
                            final_result[counter] = next(tent_seq)
                            counter +=1
                    elif switch == '1':
                        final_result[counter] = next(sin_seq)
                        counter +=1
                register_counters = 0
    return final_result
# level1 main function
def level1_final_function(seq, input_image_1d,zero_key):
    level1_seq = np.empty(len(seq), dtype=np.uint8)
    seq_256 = ((seq*(10**17)) % 256).astype('uint8')
    for i in range(len(seq)):
        if i == 0:
            level1_seq[i] = seq_256[i] ^ input_image_1d[i] ^ zero_key
        else:
            level1_seq[i] =  seq_256[i] ^ input_image_1d[i] ^ level1_seq[i-1]
    return level1_seq
def big_shift(sorted_list, image_array): # diffusion 完的圖，先做大的 row col shift
    image_array_copy = image_array.copy()
    M = len(image_array)
    N = len(image_array[0])
    for i in range(M):
        if sorted_list[i] % 2 == 0:
            shift_value = int((sorted_list[i]) % N)
            image_array_copy[i] = np.roll(image_array[i], shift = shift_value)
        else:
            shift_value = int((sorted_list[i]) % N)
            image_array_copy[i] = np.roll(image_array[i], shift = -(shift_value))
    return image_array_copy
# level2 shift main function
def shift_array(shift_seq,image_array, ca_rule):
    trans_binary = [decimal_to_binary(np.uint8(i * (10**17) % 256), 8) for i in shift_seq]
    Rshift_rules = [cell_automata(i, ca_rule) for i in trans_binary]
    Cshift_rules = [cell_automata(i, ca_rule) for i in Rshift_rules]
    sort_shift_seq = sort_list(shift_seq)
    # big row shift
    big_row_shift_subblocks = big_shift(sort_shift_seq[0:512], image_array)
    # big col shift
    big_row_rot90= np.rot90(big_row_shift_subblocks, -1)
    big_col_shift_subblocks = big_shift(sort_shift_seq[0:512], big_row_rot90)
    big_shift_subblocks = np.rot90(big_col_shift_subblocks, 1)
    # divide to subblocks
    sublocks = change_array_block8x8(big_shift_subblocks,8) # big shift 完後的 subblocks
    # 將對應sort_shift_seq排序的sublocks交換到相應的位置
    scm_sublocks = block_chages(sublocks, sort_shift_seq) # 換過位置的subblocks
    # CA rules 的 shift function
    def shift_function(sublocks, shift_rules):
        M = len(sublocks[0])
        shift_values = [int(i, 2) for i in shift_rules]
        for index in range(len(sublocks)):
            counter = 0
            for i in shift_rules[index]:
                if i == '0':
                    shift_value = int(shift_values[0] % M)
                    sublocks[index][counter] = np.roll(sublocks[index][counter], shift = shift_value)
                elif i == '1':
                    shift_value = int(shift_values[0] % M)
                    sublocks[index][counter] = np.roll(sublocks[index][counter], shift = -(shift_value))
                counter += 1
        final_blocks = sublocks
        return final_blocks
    # row shift
    Rowshift = shift_function(scm_sublocks, Rshift_rules)
    # column shift
    rot90 = [np.rot90(i, -1) for i in Rowshift]
    Colshift = shift_function(rot90, Cshift_rules)
    Colshift = [np.rot90(i, 1) for i in Colshift]
    combine_block = combine_array_block8x8(Colshift, len(Colshift))
    return combine_block
# secret key function
def secret_key(member):
    secret_key_path = r'C:\Users\Niu\Desktop\purpose_code\secret_key.txt'
    with open(secret_key_path, 'r') as file: # 從 secrets_key.txt 讀取 secret_keys
        secrets_keys = file.readlines()[:member]
    return secrets_keys
# main function
# computer_acc = 
def encode(secret_key, image_path, pic_save = False, save_path = None, set_level = None):
    input_image_1d = image_2dto1d(image_path) # 輸入圖片轉成 1D 的 array
    # parameter
    x_initial = np.float64(int(secret_key[0:12], 16) / (2**48))  
    r_values = np.float64(int(secret_key[12:28], 16) / (2**64)) *5 
    pur_item = np.int64(int(secret_key[28:], 16)+1)
    ca_rule = np.int8(pur_item % 70)
    modes = np.int8(ca_rule % 9)
    # main function 
    start_time = time.time()
    seq = switch_table((M*N), modes, x_initial, r_values, pur_item, random_rules[ca_rule]) # diffusion seq
    # level1 diffusion
    level1_pic = level1_final_function(seq, input_image_1d, ca_rule)
    # level2 confusion
    shift_seq = switch_table(4096, modes, x_initial, r_values, pur_item, random_rules[ca_rule])
    level1_pic_array = level1_pic.reshape((M,N))
    level2_shift_pic = shift_array(shift_seq, level1_pic_array.copy(), random_rules[ca_rule])
    end_time = time.time()
    print(f'exe time is {end_time - start_time} seconds')
    print("--------------------------------------------------------")
    if set_level == '1':
        show_image(level1_pic_array, pic_save= pic_save, save_path= save_path)
    elif set_level == '2':
        show_image(level2_shift_pic, pic_save= pic_save, save_path= save_path)






















