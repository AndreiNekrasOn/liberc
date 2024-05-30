import os
import numpy as np

if __name__ == "__main__":
    train_sequences = [
            'akiyo',
            'bowing',
            'bridge_close',
            'carphone',
            'container',
            'crew',
            'flower',
            'football',
            'coastguard',
            'hall_monitor',
            ]
    val_sequences = [
            'deadline',
            'bridge_far',
            'city',
            ]
    test_sequences = [
            'foreman',
            ]

    classes = np.array(range(48, 1584 + 1, 24))
    # classes = np.array([x / 100 for x in range(0 + 10, 100 + 1, 5)])

    img_dir = "images"

    # move everything to ./images/train
    for c in classes:
        files = os.listdir(f'./{img_dir}/val/{c:04d}')
        for file in files:
            os.rename(f'./{img_dir}/val/{c:04d}/{file}', f'./{img_dir}/train/{c:04d}/{file}')
        files = os.listdir(f'./{img_dir}/test/{c:04d}')
        for file in files:
            os.rename(f'./{img_dir}/test/{c:04d}/{file}', f'./{img_dir}/train/{c:04d}/{file}')
    # move all val sequences from ./{img_dir}/train/{class} to ./{img_dir}/val/{class}
    for c in classes:
        files = os.listdir(f'./{img_dir}/train/{c:04d}')
        for sequence in val_sequences:
            for file in files:
                if sequence in file:
                    os.rename(f'./{img_dir}/train/{c:04d}/{file}', f'./{img_dir}/val/{c:04d}/{file}')
        for sequence in test_sequences:
            for file in files:
                if sequence in file:
                    os.rename(f'./{img_dir}/train/{c:04d}/{file}', f'./{img_dir}/test/{c:04d}/{file}')

