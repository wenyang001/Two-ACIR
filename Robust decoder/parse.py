from struct import unpack
from collections import defaultdict
from typing import List, Tuple, Dict

# Zig-zag scan order (# https://www.w3.org/Graphics/JPEG/itu-t81.pdf, Figure A.6)
zigzag = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
          (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
          (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
          (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
          (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
          (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
          (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
          (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
    0xFFE0: "APP0"
}


class HuffmanTable:
    table_class: int
    dest_id: int
    counts: List[int]
    huff_data: Dict[Tuple[int, int], int]


class QuantizationTable:
    precision: int
    dest_id: int
    table: List[List[int]]


class FrameComponent:
    identifier: int
    sampling_factor: int
    h_sampling_factor: int
    v_sampling_factor: int
    quant_table_dest: int


class StartOfFrame:
    precision: int
    num_lines: int
    samples_per_line: int
    num_frame_components: int
    components: List[FrameComponent]


class ScanComponent:
    selector: int  # component_id 1 byte
    dc_table: int  # 1 byte - 4 bit
    ac_table: int  # 1 byte - 4 bit


class StartOfScan:
    components: List[ScanComponent]
    spectral_selection_range: Tuple[int, int]
    successive_approximation: int


def readAPP(data):
    value = unpack(">H5s2s1c2H", data[0:14])
    length, m, version, units, xdensity, ydensity = value[0], value[1], value[2], value[3], value[4], value[5]
    print(value)
    print("Section长度为{}".format(length))
    m = m.decode('utf-8')
    print("使用" + m)
    print('版本 {}.{}'.format(version[0], version[1]))
    print('密度单位 {}, x_density: {}, y_density: {}'.format(int.from_bytes(units, 'big'), xdensity, ydensity))
    return data[length:]


def readSOF(data):
    sof = StartOfFrame()
    sof.components = []
    value = unpack(">Hc2Hc", data[0:8])
    size, sof.precision, sof.num_lines, sof.samples_per_line, sof.num_frame_components = value[0], int.from_bytes(
        value[1], 'big'), value[2], value[3], int.from_bytes(value[4], 'big')
    cur_pos = 8
    for i in range(sof.num_frame_components):
        component = FrameComponent()
        t = unpack("3c", data[cur_pos:cur_pos + 3])
        cur_pos += 3
        component.identifier, component.sampling_factor, component.quant_table_dest = int.from_bytes(t[0],
                                                                                                     'big'), int.from_bytes(
            t[1], 'big'), int.from_bytes(t[2], 'big')
        component.v_sampling_factor, component.h_sampling_factor = component.sampling_factor & 0x0F, component.sampling_factor >> 4
        sof.components.append(component)

    print(f"Frame header length: {size}")
    print(f"Precision: {sof.precision}")
    print(f"Number of lines: {sof.num_lines}")
    print(f"Samples per line: {sof.samples_per_line}")
    print(f"Image size: {sof.samples_per_line} x {sof.num_lines}")
    print(f"Number of image components: {sof.num_frame_components}")
    for index, component in enumerate(sof.components):
        print(f"    Component {index + 1}: ID=0x{component.identifier:X}, "
              f"Sampling factor=0x{component.sampling_factor:X}, "
              f"Vertical sampling factor=0x{component.v_sampling_factor:X}, "
              f"Horizontal sampling factor=0x{component.h_sampling_factor:X}, "
              f"Quantization table destination=0x{component.quant_table_dest}")
    print()
    data = data[size:]
    return data, sof


def readSOS(data):
    sos = StartOfScan()
    sos.components = []
    (size,) = unpack('>H', data[0:2])
    sos.num_scan_components = int(data[2])
    cur_pos = 3
    for i in range(sos.num_scan_components):
        component = ScanComponent()
        component.selector = int(data[cur_pos])  # 颜色分量
        temp = int(data[cur_pos + 1])
        cur_pos += 2
        component.dc_table = temp >> 4
        component.ac_table = temp & 0xF
        sos.components.append(component)
        print(component.dc_table, component.ac_table)

    return data[size:], sos


def readEOI(data):
    return data


def readDHT(data: bytearray) -> List[HuffmanTable]:
    (size,) = unpack('>H', data[0:2])
    huff_tables = []
    cur_pos = 2
    while size - cur_pos > 0:
        # 一个一个去构建 HuffmanTable
        huff_table = HuffmanTable()
        huff_table.huff_data = {}

        # 低4位: Table class- 0 = DC table or lossless table, 1 = AC table
        table = int(data[cur_pos])
        cur_pos += 1
        huff_table.table_class = table >> 4

        # 高四位: Huffman Table ID - for Y & Cb/Cr
        huff_table.dest_id = table & 0xF
        huff_table.counts = [int(data[cur_pos + i]) for i in range(16)]  # 记录不同码字下的个数
        cur_pos += 16
        length_codes_map = defaultdict(list)

        # 创建范式huffcode，与标准的huffmancode不一样的点在于加了3点约束
        # 1. 最小编码长度的第一个编码必须从 0 开始。
        # 2. 相同长度编码必须是连续的。
        # 3. 编码长度为j的第一个符号可以从编码长度为j-1的最后一个符号所得知，即c_j = 2(c_{j-1}+1) -> c_j = (c_{j-1}+1) << 1 加一左移
        code = 0  # 第一条约束
        for i in range(16):
            # Go through number of codes in each length, build a map
            for j in range(huff_table.counts[i]):  # huffman 表的一个特性，同样码字的差别为1，
                huff_byte = data[cur_pos]
                huff_table.huff_data[
                    (code, i + 1)] = huff_byte  # key - value - {(code, length), data} - code就是码字 就是把code编码为data
                length_codes_map[i + 1].append(huff_byte)

                cur_pos += 1
                code += 1  # 第二条约束

            # 加一左移 #第三条约束
            code <<= 1

        print(f"Huffman table length: {size}")
        print(f"Destination ID: {huff_table.dest_id}")
        print(f"Class = {huff_table.table_class} (" +
              ("DC" if huff_table.table_class == 0 else "AC table") + ")")

        for i in range(16):
            print(f"    Codes of length {i + 1} bits ({huff_table.counts[i]} total): ", end="")
            for huff_byte in length_codes_map[i + 1]:
                print(f"{huff_byte:02X} ", end="")
            print()

        print(f"Total number of codes: {sum(huff_table.counts)}")
        huff_tables.append(huff_table)

    return data[size:], huff_tables


def readDQT(data: bytearray) -> List[QuantizationTable]:
    (size,) = unpack('>H', data[0:2])
    quant_tables = []
    cur_pos = 2
    while size - cur_pos > 0:
        quant_table = QuantizationTable()
        quant_table.table = {}

        # Create empty 8x8 table
        quant_table.table = [[0 for _ in range(8)] for _ in range(8)]

        table = int(data[cur_pos])
        cur_pos += 1

        # 低4位 precision of QT, 0 = 8 bit, otherwise 16 bit
        quant_table.precision = table >> 4
        element_bytes = 1 if quant_table.precision == 0 else 2

        # 高4位 number of Huffman table
        quant_table.dest_id = table & 0xF

        for i in range(64):
            t = 0
            for j in range(element_bytes):
                element = int(data[cur_pos])
                t = t << 8
                t += element
                cur_pos += 1
            row, col = zigzag[i]
            quant_table.table[row][col] = element

        print(f"Quantization table length: {size}")
        print("Precision: " + ("8 bits" if quant_table.precision == 0 else "16 bits"))
        print("Quantization table ID: " + str(quant_table.dest_id) +
              (" (Luminance)" if quant_table.dest_id == 0 else " (Chrominance)"))
        for i in range(len(quant_table.table)):
            print(f"    DQT, Row #{i}: " + "".join(str(element).rjust(4) for element in quant_table.table[i]))
        print()
        quant_tables.append(quant_table)

    return data[size:], quant_tables