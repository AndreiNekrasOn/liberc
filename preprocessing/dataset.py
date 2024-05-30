import numpy as np
import os
from PIL import Image
import pandas as pd
import math

SEQUENCES = [
        'akiyo',
        'bowing',
        'bridge_close',
        'bridge_far',
        'carphone',
        'city',
        'coastguard',
        'container',
        'crew',
        'deadline',
        'flower',
        'football',
        'foreman',
        'hall_monitor',
        ]
QP = range(0, 52, 20)
QPWZ = list(range(0,8,2))
CLASSES = list(range(24, 1584 + 1, 24))

def read_binary_data(filename):
    dt = np.dtype([
        ('frameNo', '>i4'),
        ('bandNo', '>i4'),
        ('level', '>i4'),
        ('length', '>i4'),
        ('source', (np.uint8, 198)),
        ('si', (np.uint8, 198))
    ])
    raw_data = np.fromfile(filename, dtype=dt)
    data = []
    for source_bytes, si_bytes in zip(raw_data['source'], raw_data['si']):
        data.append(np.bitwise_xor(source_bytes, si_bytes))
    return data, raw_data['length'] 

def convert_bitplane_to_image(data: np.ndarray, output_name='out.png', unpack=True, save=True):
    bit_matrix = data
    if unpack:
        bit_matrix = np.unpackbits(bit_matrix)
    image_size = int(np.ceil(np.sqrt(bit_matrix.shape[0])))
    pad_size = np.square(image_size) - bit_matrix.shape[0]
    bit_matrix = np.pad(bit_matrix, (0, pad_size), 'constant')
    image = bit_matrix.reshape((image_size, image_size))
    image[image == 1] = 255
    pil_image = Image.fromarray(image)
    if save == True:
        pil_image.save(output_name)
    return pil_image

def process_sequence(sequence_name, img_dir, QP, QPWZ):
    print(sequence_name)
    for qpwz in QPWZ:
        print(f'qpwz={qpwz}\nqp=[', end='')
        for qp in QP:
            df = pd.DataFrame()
            print(qp, end=', ', flush=True)
            data, lengths = read_binary_data(f'../dataset/eldat_rsi/eldat_rsi_{sequence_name}_qcif_15fps.yuv-{qp}_{qpwz}.csv')
            for i, (d, length) in enumerate(zip(data, lengths)):
                unpacked = np.unpackbits(d)
                df = pd.concat([df, pd.DataFrame({'data': [unpacked], 'length': length})])
            df['entropy'] = df['data'].apply(lambda x: computeEntropy(x))
            df.to_pickle(f'./data/{sequence_name}/{sequence_name}_{qpwz}_{qp}.pkl')
        print(']')
    print()
    return

def computeEntropy(source: np.ndarray):
    p = np.sum(source) / len(source)
    h = 0
    if p != 0 and p != 1:
        h = -p*math.log2(p)-(1-p)*math.log2(1-p)
    return h

def generate():
    for c in CLASSES:
        os.makedirs(f'./images/train/{c}', exist_ok=True)
        os.makedirs(f'./images/val/{c}', exist_ok=True)
    for sequence_name in SEQUENCES:
        os.makedirs(f'./data/{sequence_name}', exist_ok=True)
        process_sequence(sequence_name, 'train', QP, QPWZ)

if __name__ == "__main__":
    generate()

