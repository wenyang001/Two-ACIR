from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_diff(image, cur):
    diff = np.subtract(image[cur-1, :], image[cur, ])
    return np.sum(diff)

def show_cur_version1(image_arr):
    (height, width, channel) = image_arr.shape
    y = []
    for i in range(1, height-16):
        diff = np.sum(np.subtract(image_arr[i, :, :] ,  image_arr[i-1, :, :]) ** 2)
        y.append(diff)

    z = []
    for i in range(len(y)-1):
        z.append(np.absolute(y[i+1] - y[i]))

    # z = y
    plt.figure()
    fragment = []
    for i in range(len(z)):
        if z[i] > np.mean(z) * 1 and (i+2) % 16 == 0:
            fragment.append(i+2)

    if len(fragment) == 0:
        fragment.append(32)

    return fragment


def guiyi(x) -> np.ndarray:
    return (x-np.min(x))/(np.max(x)-np.min(x))


def get_z(image, fragment):
    (height, width, channel) = image.shape
    new_f = []
    print(fragment)
    for index, i in enumerate(fragment):
        k = []
        g = []
        if i == height - 16:
            for j in range(16, width, 16):
                cur = assemble_block(image, i, j)
                pre = assemble_block(image, i - 16, j)
                k.append(np.sum(np.absolute(np.subtract(cur[0, :, :], pre[15, :, :])) ** 2))

            v = np.array(k)
            pos = np.where(v == np.max(v))[0][0]
            fra_pos = pos * 16 + 16
            new_f.append([i, fra_pos])

        elif i == 16:
            for j in range(16, width, 16):
                cur = assemble_block(image, i, j)
                next = assemble_block(image, i + 16, j)
                g.append(np.sum(np.absolute(np.subtract(cur[15, :, :], next[0, :, :]))**2))

            v = np.array(g)
            pos = np.where(v == np.max(v))[0][0]
            fra_pos = pos * 16 + 16
            new_f.append([i, -fra_pos])

            val = image[:16, :, :]
            y = []

            for j in range(16, width - 16, 16):
                y.append(np.sum(np.absolute(np.subtract(val[:,j,:], val[:, j-1, :]) ** 2)))
            tmp = y.index(max(y)) * 16 + 16
            new_f.append([i, tmp])

        else:
            for j in range(16, width, 16):
                cur = assemble_block(image, i, j)
                pre = assemble_block(image, i - 16, j)
                next = assemble_block(image, i + 16, j)
                k.append(np.sum(np.absolute(np.subtract(cur[0, :, :], pre[15, :, :])) ** 2))
                g.append(np.sum(np.absolute(np.subtract(cur[15, :, :], next[0, :, :])) ** 2))

            k = np.array(k)
            g = np.array(g)
            v = np.absolute(np.array(k) - np.array(g))
            pos = np.where(v == np.max(v))[0][0]

            if np.mean(k) > np.mean(g):
                fra_pos = pos * 16 + 16
            else:
                fra_pos = -pos * 16 - 16


            val = image[i:i+16, :, :]
            y = []
            for j in range(16, width - 16, 16):
                y.append(np.sum(np.absolute(np.subtract(val[:,j,:], val[:, j-1, :]) ** 2)))

            if fra_pos > 0:
                tmp = y.index(max(y)) * 16 + 16
            else:
                tmp = -y.index(max(y)) * 16 - 16

            if abs(fra_pos) > abs(tmp):
                new_f.append([i, tmp])
                new_f.append([i, fra_pos])
            else:
                new_f.append([i, fra_pos])
                new_f.append([i, tmp])

    print("New_f:", new_f)
    return new_f

def show_partial_image(image, start, end, flag=False):
    (height, width, channel) = image.shape
    st = 0 if start[0] == 0 else start[0] - 16
    ed = height if end[0] == height else end[0] + 16
    partial = image[st:ed,:,:].copy()
    overflow = 0
    num = np.size(partial[:,:,0])
    if start[0] != 0 and end[0] != height:
        if start[1] > 0: # next
            partial[:16, :start[1], :] = overflow
            num = num-np.size(partial[:16, :start[1], 0])
        else:
            partial[:16, :, :] = overflow
            partial[16:32, :-start[1], :] = overflow
            num = num-np.size(partial[:16, :, 0])-np.size(partial[16:32, :-start[1], 0])

        if end[1] > 0: # next
            partial[-32:-16, end[1]:, :] = overflow
            partial[-16:, :, :] = overflow
            num = num-np.size(partial[-32:-16, end[1]:, 0])- np.size(partial[-16:, :, 0])
        else:
            partial[-16:, -end[1]:, :] = overflow
            num = num-np.size(partial[-16:, -end[1]:, 0])

    elif start[0] == 0 and end[0] != height:
        if end[1] > 0: # next
            partial[-32:-16, end[1]:, :] = overflow
            partial[-16:, :, :] = overflow
            num = num-np.size(partial[-32:-16, end[1]:, 0])- np.size(partial[-16:, :, 0])
        else:
            partial[-16:, -end[1]:, :] = overflow
            num = num-np.size(partial[-16:, -end[1]:, 0])

    elif start[0] != 0 and end[0] == height:
        if start[1] > 0: # next
            partial[:16, :start[1], :] = overflow
            num = num-np.size(partial[:16, :start[1], 0])
        else:
            partial[:16, :, :] = overflow
            partial[16:32, :-start[1], :] = overflow
            num = num-np.size(partial[:16, :, 0])-np.size(partial[16:32, :-start[1], 0])

    for c in range(channel):
        partial[:,:,c] = partial[:,:,c] - np.sum(partial[:,:,c]) / num
    partial = np.clip(partial, -150, 150)

    if flag:
        for c in range(channel):
            partial[:,:, c] = guiyi(partial[:,:,c]) * 255
        image_f = Image.fromarray(partial.astype('uint8'), mode='RGB')
        plt.figure()
        plt.imshow(image_f)
    return partial


def combineA_and_B(image1, image2, vec_boundary, hor_boundary):
    image = np.vstack((image1,image2))
    if hor_boundary > 0: #属于next
        image = np.delete(image, [i for i in range(vec_boundary, vec_boundary + 16)], axis = 0)
        image[vec_boundary - 16:vec_boundary, hor_boundary:, :] = image[vec_boundary:vec_boundary+16, hor_boundary:, :]
        image = np.delete(image, [i for i in range(vec_boundary, vec_boundary + 16)], axis = 0)
    else: #属于pre
        image = np.delete(image, [i for i in range(vec_boundary + 16, vec_boundary + 32)], axis = 0)
        image[vec_boundary:vec_boundary+16, -hor_boundary:, :] = image[vec_boundary+16:vec_boundary+32, -hor_boundary:, :]
        image = np.delete(image, [i for i in range(vec_boundary + 16, vec_boundary + 32)], axis = 0)
    return image


def assemble_block(image, cur_pos_x, cur_pos_y, size=16, flag=False):
    cur_block = np.concatenate((image[cur_pos_x:cur_pos_x + size, :cur_pos_y, :], image[cur_pos_x - size:cur_pos_x, cur_pos_y:, :]), axis=1)
    if flag:
        image_f = Image.fromarray(cur_block.astype('uint8'), mode='RGB')
        plt.figure()
        plt.imshow(image_f)
    return cur_block


def image_transfer(new_image_B_array):

    new_image_B_array = get_new_image(new_image_B_array, False)
    new_image_B_array = get_new_image(new_image_B_array, False)

    new_image_B_array_s = new_image_B_array.copy()
    new_image_B_array_s = reshift_image(new_image_B_array_s, False)

    return new_image_B_array, new_image_B_array_s


def reshift_image(new_image_B_array_nor, flag):
    (height, width, channel) = new_image_B_array_nor.shape
    fragment = [i for i in range(16, height - 32, 16)]
    boun = 80
    for i in range(len(fragment)):
        p = []
        cur = new_image_B_array_nor[fragment[i]:fragment[i] + 16, :, :].copy()
        pre = new_image_B_array_nor[fragment[i] - 16:fragment[i], :, :].copy()
        next = new_image_B_array_nor[fragment[i] + 16:fragment[i] + 32, :, :].copy()
        for k in range(3):
            cur[:,:,k]= cur[:,:,k] - np.mean(cur[:,:,k])
            pre[:,:,k] = pre[:,:, k] - np.mean(pre[:,:, k])
            next[:,:,k] = next[:,:, k] - np.mean(next[:,:, k])

        for j in range(-boun, boun, 16):
            if j > 0:  # append
                cur1 = np.concatenate((pre[:,width-j:,:], cur[:,:width-j,:]), axis=1)
            elif j < 0: # delete
                cur1 = np.concatenate((cur[:,-j:,:], next[:,:-j, :]), axis=1)
            else:
                cur1 = cur

            pre_diff = np.subtract(pre[15,:,0], cur1[0,:,0])**2
            next_diff = np.subtract(cur1[1,:,0], cur1[0,:,0])**2
            diff = np.sum(np.absolute(pre_diff - next_diff))
            p.append(diff)
        z = p.index(min(p)) * 16 - boun

        pos_x, pos_y = fragment[i] // 16, z // 16 * 16
        if pos_y >= 0:
            append_block = new_image_B_array_nor[fragment[i]-16:fragment[i], width-pos_y:, :]
            new_image_B_array_nor = image_insert(new_image_B_array_nor, append_block, pos_x * width)
        else:
            pos_start = pos_x * width
            pos_end = pos_start + (-pos_y)
            new_image_B_array_nor = image_delete(new_image_B_array_nor, pos_start, pos_end)

    if flag:
        plt.figure()
        image_f = Image.fromarray(new_image_B_array_nor.astype('uint8'), mode='RGB')
        plt.imshow(image_f)
    return new_image_B_array_nor


def image_insert(new_image_array, data, pos):
    (h, w, c) = new_image_array.shape
    length = 16
    V = new_image_array[0:length,:,:]
    for i in range(1, h // length):
        V = np.concatenate((V, new_image_array[i*length:i*length+length,:,:]), axis=1)
    K = np.concatenate((V[:, :pos, :], data), axis=1)
    K = np.concatenate((K, V[:,pos:,:]), axis=1)
    new_image = np.zeros((h, w, c)) + 255
    for i in range(h // length):
        new_image[i*length:i*length+length,:,:] = K[:, i*w:i*w+w, :]
    return new_image


def image_delete(new_image_array, pos_start, pos_end):
    (h, w, c) = new_image_array.shape
    length = 16
    V = new_image_array[0:length,:,:]
    for i in range(1, h // length):
        V = np.concatenate((V, new_image_array[i*length:i*length+length,:,:]), axis=1)
    K = np.concatenate((V[:, :pos_start, :],  V[:,pos_end:,:]), axis=1)
    new_image = np.zeros((h, w, c)) + 255
    (h1, w1, c1) = K.shape
    for i in range(h // length):
        if i*w + w > w1:
            new_image[i*length:i*length+length,:w1 - i*w,:] = K[:,i*w:w1, :]
        else:
            new_image[i*length:i*length+length,:,:] = K[:, i*w:i*w+w, :]
    return new_image


def subtraction_mean(new_image_array, fragment, shape):
    (h, w, c) = shape
    length = 16
    for i in range(len(fragment) - 1):
        (row, col) = fragment[i]
        (next_row, next_col) = fragment[i+1]
        cur_pos = row // 16 * w + col
        next_pos = next_row // 16 * w + next_col
        for channel in range(c):
            new_image_array[:, cur_pos:next_pos, channel] = new_image_array[:, cur_pos:next_pos, channel] - np.average(new_image_array[:, cur_pos:next_pos, channel])

        # zz = new_image_array[:, cur_pos:next_pos, channel].flatten()
        # k = [i for i in range(len(zz))]
        # plt.figure()
        # plt.scatter(k, zz)

    new_image_array = np.clip(new_image_array, -150, 150)
    new_image = np.zeros((h, w, c))
    for i in range(h // length - 1):
        new_image[i*length:i*length+length,:,:] = new_image_array[:, i*w:i*w+w, :]
    return new_image

def get_new_image(new_image_B_array, flag):
    print("Get New Image")
    (height, width, channel) = new_image_B_array.shape
    fragment = show_cur_version1(new_image_B_array)
    new_ff = get_z(new_image_B_array, fragment)

    new_ff.insert(0, [0, 0])
    new_ff.append([height, 0])
    new_image = []  # each partial image corresponding to new_ff index

    for i in range(len(new_ff) - 1):
        partial = show_partial_image(new_image_B_array, new_ff[i], new_ff[i+1], False)
        new_image.append(partial)

    k = combineA_and_B(new_image[0], new_image[1], new_ff[1][0], new_ff[1][1])  # first two

    for i in range(2, len(new_image)):
        k = combineA_and_B(k, new_image[i], new_ff[i][0], new_ff[i][1])

    for c in range(channel):
        k[:,:, c] = guiyi(k[:,:,c]) * 255

    if flag:
        draw_line(new_image_B_array, new_ff, fragment, width, height)
        image_f = Image.fromarray(k.astype('uint8'), mode='RGB')
        plt.figure()
        plt.imshow(image_f)

    return k

def draw_line(new_image_B_array, new_ff, fragment, width, height):
    plt.figure()
    plt.imshow(new_image_B_array[:,:,0],  cmap ='gray')

    plt.figure()
    plt.imshow(new_image_B_array[:,:,0],  cmap ='gray')
    for f in fragment:
        plt.axhline(f, color='b')

    plt.figure()
    plt.imshow(new_image_B_array[:,:,0],  cmap ='gray')
    for i in range(1, len(new_ff) - 1):
        h, v = new_ff[i][0], np.absolute(new_ff[i][1])
        if new_ff[i][1] > 0:
            plt.axhline(h, 0, v / width,  color="r")
            plt.axhline(h-16, v / width, 1, color='r')
            plt.axvline(v,  ymax= 1 - (new_ff[i][0] - 16) / height, ymin= 1 - (new_ff[i][0]) / height, color='r')
        else:
            plt.axhline(h+16, 0, v / width,  color="r")
            plt.axhline(h, v / width, 1, color='r')
            plt.axvline(v,  ymax= 1 - (new_ff[i][0]) / height, ymin= 1 - (new_ff[i][0] + 16) / height, color='r')





