import random

def generate_hex_string(length):
    """生成指定長度的十六進制"""
    hex_chars = '0123456789abcdef'
    return ''.join(random.choice(hex_chars) for _ in range(length))

def generate_secret_keys(filename, num_keys, key_length):
    """生成包含指定数量和長度的十六進制字符串的文件"""
    with open(filename, 'w') as f:
        for _ in range(num_keys):
            key = generate_hex_string(key_length)
            f.write(key + '\n')

# 设置文件名、键的数量和每个键的长度
filename = 'secret_key.txt'
num_keys = 100
key_length = 64

# 生成 secret_key.txt 文件
generate_secret_keys(filename, num_keys, key_length)

print(f'{num_keys} 個長度為 {key_length} 的十六進制字符串已生成到 {filename} 文件中。')
