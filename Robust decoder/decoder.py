
from struct import unpack
import math
import parse
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from alignment import image_transfer

debug = False

marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",

}

def get_signed_value(bits: int, num_bits: int) -> int:
    if bits < 2**(num_bits - 1):  # Check if bits is less than the middle value
        min_val = (-1 << num_bits) + 1
        return min_val + bits

    return bits

# Stream class for reading bits from a byte array
class Stream:
    def __init__(self, data):
        # self.bindata = BitArray(hex=data.hex())
        self.data = data
        self.pos = 0
        self.length = len(data) * 8
        self.end = False

    def GetBit(self):
        if self.pos > self.length - 1:
            self.end = True
            return 0
        b = self.data[self.pos >> 3]
        s = 7 - (self.pos & 0x7)
        self.pos += 1
        return (b >> s) & 0x1

    def GetBitN(self, l):
        val = 0
        for _ in range(l):
            val = val * 2 + self.GetBit()
        return val

class JpegDecoder(object):
    def __init__(self):
        self.quant_tables = {}
        self.huffman_tables = {}
        self.sof = None
        self.sos = None
        self.num_mcu_x = 0
        self.num_mcu_y = 0
        self.mcu_size_x = 0
        self.mcu_size_y = 0
        self.decoding_error = 0
        self.zero_mcu_array = None

    def ycbcr_to_rgb(self, ycbcr):
        # R  = Y +                       + (Cr - 128) *  1.40200
        # G  = Y + (Cb - 128) * -0.34414 + (Cr - 128) * -0.71414
        # B  = Y + (Cb - 128) *  1.77200
        m = np.array([[ 1.0,      1.0,      1.0],
                    [-0.0,     -0.34414,  1.772],
                    [ 1.40200, -0.71414 , 0.0] ])
        rgb = ycbcr.dot(m) + 128
        return rgb

    def get_next_huffman_value(self, scanStream: Stream, huff_table: [Dict[Tuple[int, int], int], int], DCorAC,
                           debug_print=False) -> Tuple[int, int]:       # 如果结束返回0，每次都是会带入 (0, 2) 查表
        encoded_bits = 0
        num_bits = 0

        while (encoded_bits, num_bits) not in huff_table[0]:
            if debug_print:
                if DCorAC:
                    print('aC:', encoded_bits, num_bits)
                else:
                    print('dC:', encoded_bits, num_bits)

            if scanStream.end:
                return 0, True

            # update (encoded_bits, num_bits)
            encoded_bits = encoded_bits * 2 + scanStream.GetBit()
            num_bits += 1

            if num_bits > huff_table[1]:  # max length of huffman table
                self.decoding_error += 1
                return 0, True

        return huff_table[0][(encoded_bits, num_bits)], False

    def removeFF00(self, data):
        datapro = bytearray()
        i = 0
        while i < len(data) - 2:
            b, bnext = unpack("BB", data[i : i + 2])
            if b == 0xFF:
                if bnext != 0x00:
                    print("ocurring non 0xFF00", hex(b), hex(bnext))
                    i += 2
                    continue
                datapro.append(data[i])
                i += 2
            else:
                datapro.append(data[i])
                i += 1
        return datapro

    def read_dc(self, scanStream, dc_huff_table):  # (4,0) & (2, 0)
        codelen, FLAG = self.get_next_huffman_value(scanStream, dc_huff_table, DCorAC = 0, debug_print=False)

        if FLAG == True:
            return 0, True
        else:
            additional_bits = scanStream.GetBitN(codelen)
            diff = get_signed_value(additional_bits, codelen)

        return diff, False  # False

    def read_ac(self, dct_coeffs, scanStream, ac_huff_table):
        k = 1
        FLAG = False
        while k < 64:
            rs, FLAG = self.get_next_huffman_value(scanStream, ac_huff_table,  DCorAC = 1, debug_print=False)
            if FLAG == True:
                break

            # ac decoding
            ssss = rs & 0x0F  #(rrrr, ssss) = (runlength, codelen)
            rrrr = rs >> 4
            if ssss == 0:
                if rrrr == 0:  # end of block
                    break
                elif rrrr == 0x0F:
                    k += 16
            else:
                k += rrrr

                if k >= 64:
                    FLAG = True
                    self.decoding_error += 1
                    break

                additional_bits = scanStream.GetBitN(ssss)
                dct_coeffs[k] = get_signed_value(additional_bits, ssss)
                k += 1

        return dct_coeffs, FLAG


    def decodeMCU(self, scanStream, predictions):  # 4个huffman table - YYYY Cb Cr
        mcu_arr = np.zeros((self.mcu_size_y, self.mcu_size_x, 3))

        for component_idx, component in enumerate(self.sos.components):
            frame_component = None
            for c in self.sof.components:
                if c.identifier == component.selector:
                    frame_component = c
                    break

            assert frame_component is not None
            dc_huff_table = self.huffman_tables[frame_component.identifier // 2, 0]
            ac_huff_table = self.huffman_tables[frame_component.identifier // 2, 1]

            # Initialize 2D array for MCU， 每个 MCU为 8*8 或者 16*16 = mcu = [[64],[64],...]
            mcu = np.zeros((8 * frame_component.v_sampling_factor, 8 * frame_component.h_sampling_factor))

            # print(mcu.shape, component_idx, component.selector)

            # YCbCr 4:4:4 or YYYYCbCr 4:2:0
            for data_unit_row in range(frame_component.v_sampling_factor):
                for data_unit_col in range(frame_component.h_sampling_factor):

                    # initialize 2D array for dct coefficients
                    dct_coeffs = [0 for _ in range(64)]

                    # Huffman decoding of DC coefficient
                    diff, dc_error_FLAG = self.read_dc(scanStream, dc_huff_table)
                    if dc_error_FLAG == True:
                        return None, True

                    dct_coeffs[0] = predictions[component_idx] + diff
                    predictions[component_idx] = dct_coeffs[0]

                    # Huffman decoding of AC coefficients
                    dct_coeffs, ac_error_FLAG = self.read_ac(dct_coeffs, scanStream, ac_huff_table)

                    if ac_error_FLAG == True:
                        return None, True

                    # Get quantization table
                    quant_table = self.quant_tables[frame_component.quant_table_dest]
                    dct_matrix = np.zeros((8, 8))
                    for i, coeff in enumerate(dct_coeffs):
                        row, col = parse.zigzag[i]
                        dct_matrix[row][col] = coeff * quant_table[row][col]

                    # IDCT
                    block = cv2.idct(dct_matrix)
                    mcu[data_unit_row * 8: data_unit_row * 8 + 8, data_unit_col * 8: data_unit_col * 8 + 8] = block

            # Expand MCU to maximum MCU size by duplicating values vertically or horizontally
            horiz_multiplier = self.mcu_size_x // 8 // frame_component.h_sampling_factor
            vert_multiplier = self.mcu_size_y // 8 // frame_component.v_sampling_factor

            if vert_multiplier > 1 or horiz_multiplier > 1:
                mcu_list = mcu.tolist()
                mcu_expand = np.array([[val for val in row for _ in range(horiz_multiplier)]
                                   for row in mcu_list for _ in range(vert_multiplier)])
                mcu_arr[:, :, component_idx] = mcu_expand
            else:
                mcu_arr[:, :, component_idx] = mcu

        return mcu_arr, False  # False means no error

    def get_maxlength_HT(self, HT):
        v = []
        for key in HT.keys():
            v.append(key[1])
        return max(v)

    def decode_image(self, data):
        while True:
            (marker,) = unpack(">H", data[0:2])
            print(marker_mapping.get(marker))
            if marker == 0xFFD8: # begin
                data = data[2:]

            elif marker == 0xFFD9:  # end
                return

            elif marker >= 0xFFE0 and marker <= 0xFFEF:
                data = parse.readAPP(data[2:])

            elif marker == 0xFFC0 or marker == 0xFFC1:
                data, self.sof = parse.readSOF(data[2:])
                max_h_sampling_factor = 0
                max_v_sampling_factor = 0

                for component in self.sof.components:
                    max_h_sampling_factor = max(max_h_sampling_factor, component.h_sampling_factor)
                    max_v_sampling_factor = max(max_v_sampling_factor, component.v_sampling_factor)

                self.mcu_size_x = 8 * max_h_sampling_factor
                self.mcu_size_y = 8 * max_v_sampling_factor

                self.num_mcu_x = math.ceil(self.sof.samples_per_line / self.mcu_size_x)
                self.num_mcu_y = math.ceil(self.sof.num_lines /self.mcu_size_y)

                self.zero_mcu_array = np.zeros((self.mcu_size_y, self.mcu_size_x, len(self.sof.components)))

            # Huffman table
            elif marker == 0xFFC4:
                data, HTs = parse.readDHT(data[2:])
                for table in HTs:
                    num = self.get_maxlength_HT(table.huff_data)
                    self.huffman_tables[table.dest_id, table.table_class] = [table.huff_data, num]

            # Quantization table
            elif marker == 0xFFDB:
                data, QTs = parse.readDQT(data[2:])
                for table in QTs:
                    self.quant_tables[table.dest_id] = table.table

            elif marker == 0xFFDA:
                data, self.sos = parse.readSOS(data[2:])
                assert self.sof is not None
                data = self.removeFF00(data)
                scanStream = Stream(data)
                predictions = [0 for _ in range(len(self.sos.components))]
                numpy_rgb = np.zeros((self.mcu_size_y * self.num_mcu_y,  self.mcu_size_x * self.num_mcu_x, len(self.sos.components)), dtype=np.float64)

                for mcu_row in range(self.num_mcu_y):
                    mcu_col = 0
                    while mcu_col < self.num_mcu_x:
                        pos_backup = scanStream.pos
                        predictions_backup = predictions[:]
                        MCU_rgb, error_FLAG = self.decodeMCU(scanStream, predictions)

                        # print(scanStream.pos, scanStream.end)
                        if error_FLAG == True:  # decoding fails or end of image
                            if scanStream.end:
                                y_pos, x_pos = mcu_row * self.mcu_size_y, mcu_col * self.mcu_size_x
                                numpy_rgb[y_pos:y_pos + self.mcu_size_y, x_pos: x_pos + self.mcu_size_x, :] = self.zero_mcu_array
                                mcu_col += 1

                            else:
                                scanStream.pos = pos_backup + 1
                                predictions = predictions_backup[:]
                                continue

                        else:  # decoding succeeds
                            y_pos, x_pos = mcu_row * self.mcu_size_y, mcu_col * self.mcu_size_x
                            numpy_rgb[y_pos:y_pos + self.mcu_size_y, x_pos: x_pos + self.mcu_size_x, :] = MCU_rgb
                            mcu_col += 1

                numpy_rgb = self.ycbcr_to_rgb(numpy_rgb)

                # alignment
                numpy_rgb, numpy_rgb_s = image_transfer(numpy_rgb)

                for c in range(3):
                    numpy_rgb[:, :, c] = self.guiyi(numpy_rgb[:, :, c]) * 255

                numpy_rgb = numpy_rgb[:self.sof.num_lines, : self.sof.samples_per_line, :]
                numpy_rgb_s = numpy_rgb_s[:self.sof.num_lines, : self.sof.samples_per_line, :]
                image, image_s = Image.fromarray(numpy_rgb.astype('uint8')), Image.fromarray(numpy_rgb_s.astype('uint8'))
                return image, image_s

            else:  # All other segments have length specified at the start, skip for now
                data = data[2:]
                (size,) = unpack('>H', data[0:2])
                data = data[size:]

    def guiyi(self, x) -> np.ndarray:
        return (x-np.min(x))/(np.max(x)-np.min(x))

def decode(file):
    file_path = decode_path + r'/' + file
    print(file_path)
    with open(file_path, "rb") as f:
        image, image_s = decoder.decode_image(f.read())
        image.save(decode_path[:-3] + r'/Input/' + file.replace('.jpg', '.png'))
        image_s.save(decode_path[:-3] + r'/Input_S/' + file.replace('.jpg', '.png'))
        print('Finish')


def get_file_list(file_path):
    files_list = []
    for root, dirs, files in os.walk(file_path):
        for filename in files:
            files_list.append(filename)
    return files_list


if __name__ == '__main__':
    import os
    from multiprocessing import Pool
    decoder = JpegDecoder()
    decode_path = r'../Datasets/Cityscape2K/De/'
    decode_files = get_file_list(decode_path)

    pool = Pool(4)
    pool.map(decode, decode_files)
    pool.close()
    pool.join()
