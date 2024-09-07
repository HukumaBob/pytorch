from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

for i in range(10):
    # Создаем изображение 28x28 с черным фоном
    img = Image.new('L', (28, 28), color=0)  # 'L' для градаций серого, черный фон

    # Рисуем цифру
    draw = ImageDraw.Draw(img)
    
    try:
        # Загружаем шрифт, если он доступен
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except IOError:
        # Используем стандартный шрифт, если загрузка не удалась
        font = ImageFont.load_default()

    draw.text((4, 0), str(i), font=font, fill=255)  # Рисуем цифру, fill=255 - белый цвет

    # Преобразуем изображение в массив и визуализируем
    img_array = np.array(img)
    plt.imshow(img_array, cmap='gray')
    plt.title(f'Image {i}')
    plt.show()

    # Сохранение изображения
    file_name = str(i) + '.png'
    img.save(file_name)
