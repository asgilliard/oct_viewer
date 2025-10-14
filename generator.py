import numpy as np


def generate_dat(shape=(512, 512, 512), filename='images/generated.dat'):
    Y, Z, X = np.ogrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    Z = (Z - shape[0] // 2).astype(np.float32)
    Y = (Y - shape[1] // 2).astype(np.float32)
    X = (X - shape[2] // 2).astype(np.float32)

    # Radius
    radius = np.sqrt(X**2 + Y**2 + Z**2)

    # Rings + dim
    volume = np.sin(radius / 10) * np.exp(-radius / 200) * 127 + 128
    volume = np.clip(volume, 0, 255).astype(np.uint8)

    volume.tofile(filename)
    print(f'Generated {filename} ({shape[0]}×{shape[1]}×{shape[2]})')


if __name__ == '__main__':
    generate_dat()
