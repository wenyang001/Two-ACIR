# -*- coding: UTF-8 -*-

from android_aes import AES_CBC_ESSIV
import os
import random
import numpy as np
import pickle
from multiprocessing import Pool

data, encrypted_data, decrypted_data = b'', b'', b''
key = b'"\x81\xf4\xba\xef\xa5:\x89fWtP\xc7\x92o\xbf'  # key 的长度为16个byte
cipher = AES_CBC_ESSIV(key)
PATH = os.getcwd() + '/'
BLOCK_SIZE = 512

image_path = PATH + r'../Datasets/Cityscape2K/'
data_path = image_path + r"GT"
en = image_path + r"En"
de = image_path + r"De"


def bytecompare(byte_a, byte_b, width):
    tmp = [1] * len(byte_a)
    for i in range(len(byte_a)):
        if byte_a[i] == byte_b[i]:
            tmp[i] = 1  # True
        else:
            tmp[i] = 0  # False

    height = len(byte_a) // width + 1
    result = [[1] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            if i * width + j < len(byte_a):
                result[i][j] = tmp[i * width + j]
            else:
                result[i][j] = 'x'

    np.savetxt('bytecompare.txt', result, fmt='%.4s')

    for i in range(height):
        print(result[i])


# bytes = 8 bits = two hex
def bit_flip(byte_array, block_id, byte_num, bit_num):
    byte_num = byte_num + block_id * 512
    if bit_num < 8:
        bits = '{:08b}'.format(byte_array[byte_num])
        if bits[bit_num] == '1':
            bits = bits[:bit_num] + '0' + bits[bit_num + 1:]
        else:
            bits = bits[:bit_num] + '1' + bits[bit_num + 1:]

        # bytes stored in new_byte_array is interger
        byte_array[byte_num] = int(bits, 2)


def encrypt(file):
    with open(data_path + '/' + file, 'rb') as f:
        image = f.read()
        f.close()

    with open(en + '/' + file, "wb") as f:
        cipher.encrypt(image, file_output=f)
        f.close()


def decrypt(file, errors):
    with open(en + '/' + file, "rb") as f:
        encrypted_data = f.read()
        f.close()

    encrypted_data_array = bytearray(encrypted_data)

    # add error
    for error in errors:
        bit_flip(encrypted_data_array, error[0], error[1], error[2])

    # decryption
    with open(de + '/' + file, 'wb') as f:
        cipher.decrypt(bytes(encrypted_data_array), file_output=f)
        f.close()


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def readImage(IMAGE_TO_OPEN):
    start_pos, end_pos = 0, 0
    with open(IMAGE_TO_OPEN, "rb") as f:
        block_id_bytes = f.read(2)
        while block_id_bytes:
            block_id = int.from_bytes(block_id_bytes, byteorder="big")
            if block_id == 0xFFD9:    # -------------------------------------- End of Image (EOI)
                break

            elif block_id == 0xFFDA:  # -------------------------------------- Start of Scan (SOS)
                size = int.from_bytes(f.read(2), byteorder="big")
                f.seek(size - 2, 1)

                # JPEG raw data
                start_pos = f.tell()
                f.seek(-2, 2)
                end_pos = f.tell()

            elif block_id == 0xFFD8:  # -------------------------------------- Start of Image (SOI)
                pass

            else:  # All other segments have length specified at the start, skip for now
                size = int.from_bytes(f.read(2), byteorder="big")
                f.seek(size - 2, 1)

            block_id_bytes = f.read(2)

        return start_pos, end_pos

def error_generate(BER):
    image_dic = {}
    print(data_path)
    if not os.path.exists(image_path + 'image.pkl'):
        for root, dirs, files in os.walk(data_path):
            for index, file in enumerate(files):
                print(file)
                dic = {'name': [], 'start': [], 'end': [], 'error': []}
                path = os.path.join(root, file)
                (start, end) = readImage(path)
                start_block, end_block = int(start / BLOCK_SIZE) + 1, int(end / BLOCK_SIZE)
                dic['name'].append(file)
                dic['start'].append(start)
                dic['end'].append(end)
                for block_id in range(start_block, end_block):
                    for bytenum in range(BLOCK_SIZE):  # 400kB = 400 * 1000 * 8 = 3,200,000 % 10000 = 32
                        for bitnum in range(8):
                            if random.random() < BER:  # error
                                dic['error'].append((block_id, bytenum, bitnum))
                image_dic[index] = dic
            save_dict(image_dic, image_path + 'image')
    else:
        print('exist')


def inject_error(key):
    encrypt(image_dic[key]['name'][0])
    decrypt(image_dic[key]['name'][0], image_dic[key]['error'])


if __name__ == '__main__':
    BER = pow(10, -5)

    # 1st randomly generate the bit error position for every JPEG file and store in image.pkl
    error_generate(BER=BER)

    # 2nd load the image.pkl
    image_dic = load_dict(image_path + 'image')

    # 3rd encrypt the JPEG file and inject bit error, and then decrypt the JPEG file with bit errors.
    pool = Pool(4)
    pool.map(inject_error, image_dic.keys())
    pool.close()
    pool.join()




