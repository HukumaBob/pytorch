# Объяснение:
#     Сверточный слой (функция convolve2d):
#         Мы используем фильтр размером 3x3, который применяет свертку ко всему изображению.
#         Шаг свертки равен 1, что означает, что фильтр двигается по одному пикселю за раз.
#         Для каждого положения фильтра вычисляется сумма произведений элементов фильтра на 
#         соответствующие элементы изображения, после чего результат записывается в выходную матрицу.

#     Max Pooling (функция max_pooling):
#         Мы используем окно размером 2x2 для pooling, чтобы уменьшить размер выходного изображения.
#         В каждом 2x2 блоке мы выбираем максимальное значение, что позволяет уменьшить размерность, 
#         сохраняя наиболее важную информацию.

# Вывод:

# Этот код показывает базовую работу сверточной нейросети с использованием свертки и max pooling. 
# Хотя на практике для реализации CNN обычно используются высокоуровневые библиотеки (например, TensorFlow, PyTorch), 
# этот пример иллюстрирует основные принципы работы на уровне numpy.

import numpy as np

# Пример изображения 5x5
image = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 0],
    [7, 8, 9, 2, 1],
    [0, 1, 2, 3, 4],
    [3, 0, 1, 2, 5]
])

# Ядро 3x3 для свертки (например, фильтр для выявления границ)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Размер шага свертки (stride)
stride = 1

# Функция для выполнения свертки
def convolve2d(image, kernel, stride):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    
    # Рассчитываем размеры выходного изображения
    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width))
    
    # Применение свертки
    for i in range(0, output_height):
        for j in range(0, output_width):
            region = image[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Применяем свертку
conv_output = convolve2d(image, kernel, stride)

print("Результат свертки:")
print(conv_output)

# Функция для max pooling (пуллинг 2x2 с шагом 2)
def max_pooling(image, pool_size, stride):
    pool_height, pool_width = pool_size
    image_height, image_width = image.shape
    
    # Рассчитываем размеры выходного изображения
    output_height = (image_height - pool_height) // stride + 1
    output_width = (image_width - pool_width) // stride + 1
    output = np.zeros((output_height, output_width))
    
    # Применение max pooling
    for i in range(0, output_height):
        for j in range(0, output_width):
            region = image[i*stride:i*stride + pool_height, j*stride:j*stride + pool_width]
            output[i, j] = np.max(region)
    
    return output

# Применяем max pooling
pool_output = max_pooling(conv_output, pool_size=(2, 2), stride=2)

print("\nРезультат max pooling:")
print(pool_output)
