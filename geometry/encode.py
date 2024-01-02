import numpy as np
from PIL import Image
import math
import secrets
import time
M,N = 512,512
image_path = '../picture/image/t001.png'

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
def decimal_to_binary(decimal_num):
    binary_str = bin(decimal_num)[2:]  # 將十進制轉換為二進制，並去掉前綴'0b'
    num_zeros_to_add = 64 - len(binary_str)  # 計算需要補零的位數
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
def switch_table(M,N, modes, r_values, x_initial, pur_item, ca_rule):
    # 控制流量
    control_bits = 64
    iterations = M*N
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
                trans_binary = decimal_to_binary(np.uint64(next(logistic_seq) * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(next(tent_seq) * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(next(sin_seq) * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)))
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
                trans_binary = decimal_to_binary(np.uint64(register * (10**20) % (2**64)))
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




member = 1 # 100 組使用者
for i in range(member):
    random_number = secrets.token_hex(16)
    x_initial = np.float64(int(random_number[0:12], 16) / (2**48))  
    r_values = np.float64(int(random_number[12:28], 16) / (2**64)) *5
    pur_item = np.int64(int(random_number[28:], 16))
    ca_rule = np.int8((int(random_number[28:], 16)) % 70)
    modes = ca_rule % 9

for j in range(9):
    start_time = time.time()
    final_result = switch_table(M,N,j,x_initial, r_values, pur_item, random_rules[ca_rule])
    print(len(final_result))
    print(type(final_result[0]))
    print(final_result)
    print(final_result[::-1])
    end_time = time.time()
    print(f'exe time is {end_time - start_time} seconds')
    print("--------------------------------------------------------")










































# if __name__ == '__main__':
    # member = 100 # 3 組
    # for i in range(member):
    #     random_number = secrets.token_hex(16)
    #     x_initial = int(random_number[0:12], 16) / (2**48)  # Initial value of x
    #     r_values = (int(random_number[12:28], 16) / (2**64)) *5
    #     pur_item = int(random_number[28:], 16)
    #     ca_rule = (int(random_number[28:], 16)) % 70
    #     final_result = (interate(x_initial, r_values, pur_item, ca_rule))
        # file_name = os.path.join(f"randomness_testsuite/yuhidata/{i}.txt")
        # with open(file_name, 'w') as file:
        #     for result in final_result:
        #         binary_result = decimal_to_binary(int(result*(2**24)))
        #         file.write(f"{binary_result}\n")