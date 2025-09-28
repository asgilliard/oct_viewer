import numpy as np

# Создаём градиент 512x512
data = np.linspace(0, 255, 512*512, dtype=np.uint8).reshape((512, 512))

# Сохраняем в dat-файл
data.tofile("example.dat")
