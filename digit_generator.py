from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

for i in range(10):
    # Создаем изображение 28x28 с белым фоном
    img = Image.new('L', (28, 28), color=255)  # 'L' для градаций серого

    # Рисуем цифру
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)  # Можно выбрать любой шрифт
    draw.text((4, 0), str(i), font=font, fill=0)  # Рисуем цифру, fill=0 - черный цвет

    # Преобразуем изображение в массив и визуализируем
    img_array = np.array(img)
    plt.imshow(img_array, cmap='gray')
    plt.show()

    # Сохранение изображения
    file_name = 'test' + str(i) + '.png'
    img.save(file_name)