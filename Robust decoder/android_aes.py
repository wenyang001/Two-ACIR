# -*- coding: UTF-8 -*-
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from hashlib import sha256
from struct import pack
import math

block_size_k = 512  # 设置磁盘block的size为512 bytes


class AES_CBC_ESSIV:
    def __init__(self, key):
        self.key = key
        self.salt = sha256(self.key).digest()
        self.essiv_cipher = AES.new(self.salt, AES.MODE_ECB)

    def decrypt(self, data, file_output=b'', block_size=block_size_k):
        decrypted_data = b''
        number_of_blocks = int(math.ceil(float(len(data)) / block_size))

        for block_number in range(number_of_blocks):
            long_block_number = pack(">I",
                                     block_number) + b"\x00" * 12  # pack函数用来进行打包，<表示little-endian， I表示unsigned int,一个unsigned int = 4 byte
            essiv = self.essiv_cipher.encrypt(long_block_number)
            cipher = AES.new(self.key, AES.MODE_CBC, essiv)
            decrypted_data += cipher.decrypt(data[block_number * block_size: (block_number + 1) * block_size])

        decrypted_data = unpad(decrypted_data, 16)
        if file_output != b'':
            file_output.write(decrypted_data)

        return decrypted_data

    def encrypt(self, data, file_output=b'', block_size=block_size_k):
        data = pad(data, 16)
        encrypted_data = b''
        number_of_blocks = int(math.ceil(float(len(data)) / block_size))

        # print(len(data), number_of_blocks)

        for block_number in range(number_of_blocks):
            long_block_number = pack(">I", block_number) + b"\x00" * 12
            essiv = self.essiv_cipher.encrypt(long_block_number)
            cipher = AES.new(self.key, AES.MODE_CBC, essiv)
            encrypted_data += cipher.encrypt(data[block_number * block_size: (block_number + 1) * block_size])
            # print(len(data[block_number * block_size: (block_number + 1) * block_size]))
        if file_output != b'':
            file_output.write(encrypted_data)
        return encrypted_data
