from PIL import Image

def resize_512(file_name):
    scale = 4

    file_path = path['GT'] + file_name
    file = Image.open(file_path).convert('RGB')
    file_name = file_name.replace('.jpg', '.png')

    (ow, oh) = file.size
    resized_file = file.resize((ow // scale, oh // scale), Image.BICUBIC)
    resized_file.save(path_512['GT'] + file_name)

    file_path = path['Input'] + file_name
    file = Image.open(file_path).convert('RGB')
    (ow, oh) = file.size
    resized_file = file.resize((ow // scale, oh // scale), Image.BICUBIC)
    resized_file.save(path_512['Input'] + file_name)

    file_path = path['Input_S'] + file_name
    file = Image.open(file_path).convert('RGB')
    (ow, oh) = file.size
    resized_file = file.resize((ow // scale, oh // scale), Image.BICUBIC)
    resized_file.save(path_512['Input_S'] + file_name)

def resize_1K(file_name):
    scale = 2

    file_path = path['GT'] + file_name
    file = Image.open(file_path).convert('RGB')
    file_name = file_name.replace('.jpg', '.png')
    (ow, oh) = file.size
    resized_file = file.resize((ow // scale, oh // scale), Image.BICUBIC)
    resized_file.save(path_1K['GT'] + file_name)

    file_path = path['Input'] + file_name
    file = Image.open(file_path).convert('RGB')
    (ow, oh) = file.size
    resized_file = file.resize((ow // scale, oh // scale), Image.BICUBIC)
    resized_file.save(path_1K['Input'] + file_name)

    file_path = path['Input_S'] + file_name
    file = Image.open(file_path).convert('RGB')
    (ow, oh) = file.size
    resized_file = file.resize((ow // scale, oh // scale), Image.BICUBIC)
    resized_file.save(path_1K['Input_S'] + file_name)



def get_file_list(file_path):
    files_list = []
    for root, dirs, files in os.walk(file_path):
        for filename in files:
            files_list.append(filename)
    return files_list


if __name__ == '__main__':
    import os
    from multiprocessing import Pool

    path = {
        'GT': r'../Datasets/Cityscape2K/GT/',
        'Input': r'../Datasets/Cityscape2K/Input/',
        'Input_S': r'../Datasets/Cityscape2K/Input_S/',
    }

    path_512 = {
        'GT': r'../Datasets/test512/GT/',
        'Input': r'../Datasets/test512/Input/',
        'Input_S': r'../Datasets/test512/Input_S/',
    }

    path_1K = {
        'GT': r'../Datasets/test1K/GT/',
        'Input': r'../Datasets/test1K/Input/',
        'Input_S': r'../Datasets/test1K/Input_S/',
    }


    for key in path.keys():
        if not os.path.exists(path[key]):
            os.makedirs(path[key])
        if not os.path.exists(path_512[key]):
            os.makedirs(path_512[key])
        if not os.path.exists(path_1K[key]):
            os.makedirs(path_1K[key])


    decode_files = get_file_list(path['GT'])
    print(decode_files)

    pool = Pool(4)
    pool.map(resize_512, decode_files)
    pool.map(resize_1K, decode_files)
    pool.close()
    pool.join()

