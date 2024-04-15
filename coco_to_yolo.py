import os
import json
import shutil
import random
import click

import yaml
from shapely.geometry import Polygon
from shapely.affinity import rotate

from ImageElement import ImageElement


def preprocessing_for_yolov8_obb_model(coco_json: str, language_ru=False):
    """
    Checks for Oriented Bounding Boxes in COCO format. If found,
    replaces the bbox and rotation of each object with the coordinates of four points in the segmentation section.
    
    Args:
    - coco_json (str): Path to the file containing COCO data in JSON format.
    - language_ru (bool): If True, all comments will be in Russian (otherwise in English).
    """

    # Loading COCO data from file 
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # Getting the list of annotations from COCO
    annotations = coco_data['annotations']
    changes = 0

    # Iterating through the annotations
    for annotation in annotations:
        segmentation = annotation['segmentation']

        # If segmentation is empty and bbox contains information, perform the operation
        if not segmentation and annotation['bbox']:
            bbox = annotation['bbox']
            rotation_angle = annotation['attributes']['rotation']  # Assumes rotation information is available

            # Converting bbox to x, y, width, height format
            x, y, width, height = bbox

            # Creating a rotated rectangle
            rectangle = Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])

            # Rotating the rectangle
            rotated_rectangle = rotate(rectangle, rotation_angle, origin='center')

            # Getting the coordinates of the vertices of the rotated rectangle
            new_segmentation = list(rotated_rectangle.exterior.coords)

            # Keeping only the vertex coordinates (first 4 elements)
            new_segmentation = new_segmentation[:4]

            # Converting the list of vertices into the desired format
            flattened_segmentation = [coord for point in new_segmentation for coord in point]

            # Updating the value in the annotation
            annotation['segmentation'] = [flattened_segmentation]

            changes += 1

    if changes > 0:
        if language_ru:
            print(f'Было обнаружено {changes} Oriented Bounding Boxes в файле {coco_json}')
        else:
            print(f'Found {changes} Oriented Bounding Boxes in the file {coco_json}')

        # Saving the updated data to the file
        with open(coco_json, 'w') as f:
            json.dump(coco_data, f)


@click.command()
@click.option(
    "--coco_dataset",
    default="COCO_dataset",
    help="Папка с датасетом формата COCO 1.0 (можно выгрузить из CVAT). По умолчанию COCO_dataset",
    type=str,
)
@click.option(
    "--yolo_dataset",
    default="YOLO_dataset",
    help="Папка с итоговым датасетом формата YOLOv8. По умолчанию YOLO_dataset",
    type=str,
)
@click.option(
    "--print_info",
    default=False,
    help="Вкл/Выкл режима вывода логов обработки. По умолчанию отключен",
    type=bool,
)
@click.option(
    "--autosplit",
    help="Вкл/Выкл режима автоматического разделения на train/val. По умолчанию отключен (берет согласно разметке CVAT)",
    default=False,
    type=bool,
)
@click.option(
    "--percent_val",
    help="Процент данных на val при выборе режима autosplit=True. По умолчанию 25%",
    default=25,
    type=float,
)
def main(**kwargs):
    # ------------------ ARG parse ------------------
    coco_dataset_path = kwargs["coco_dataset"]
    yolo_dataset_path = kwargs["yolo_dataset"]
    print_info = kwargs["print_info"]
    autosplit = kwargs["autosplit"]
    percent_val = kwargs["percent_val"]

    coco_annotations_path = os.path.join(coco_dataset_path, 'annotations')
    coco_images_path = os.path.join(coco_dataset_path, 'images')

    # Проверяем наличие датасета
    if not os.path.exists(coco_dataset_path):
        raise FileNotFoundError(f"Папка с COCO датасетом '{coco_images_path}' не найдена. ")


    # Проверяем наличие папки с изображениями
    if not os.path.exists(coco_images_path):
        raise FileNotFoundError(f"Папка с изображениями '{coco_images_path}' не найдена. "
                                f"Убедитесь, что вы загрузили разметку COCO так, чтобы имелась папка со всеми изображениями.")

    # Проверяем, существует ли папка annotations
    if not os.path.exists(coco_annotations_path):
        raise FileNotFoundError(f"Папка с json файлами '{coco_annotations_path}' не найдена.")


    list_of_image_elements = []
    list_of_images_path = []

    # Получаем список всех файлов в папке annotations
    annotation_files = os.listdir(coco_annotations_path)

    shutil.rmtree(yolo_dataset_path, ignore_errors=True) # очищаем старые данные в папке

    if autosplit:
        for folder_path in ['images', 'labels']:
            for type in ['validation', 'train']:
                path_create=os.path.join(yolo_dataset_path, type, folder_path)
                os.makedirs(path_create, exist_ok=True)

    ### Запуск проверки отсутвия дубликатов в разных подвыборках ###
    # Создаем словарь для хранения файлов и их соответствующих JSON файлов
    file_json_mapping = {}

    # Проходим по файлам аннотаций
    for annotation_file in annotation_files:
        json_file_path = os.path.join(coco_annotations_path, annotation_file)
        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        # Получаем список изображений из JSON
        images = coco_data['images']

        # Проходим по изображениям и обновляем словарь file_json_mapping
        for image in images:
            file_name = image['file_name']
            if file_name not in file_json_mapping:
                file_json_mapping[file_name] = [annotation_file]
            else:
                file_json_mapping[file_name].append(annotation_file)

    # Проверяем, если у какого-либо файла есть более одного вхождения
    for file_name, json_files in file_json_mapping.items():
        if len(json_files) > 1:
            print(f"Файл {file_name} встречается в следующих JSON файлах: {json_files}")
            print(f'В каком-либо из JSON файлов удалите в разделе "images" словарь ' \
                  f'с описанием этой фотографии, иначе будет ошибка при выполнении кода')
            raise SystemExit

    ### Запуск оновного кода ###
           
    # Проходим по файлам аннотаций
    for annotation_file in annotation_files:
        # Парсим название файла изображения из файла аннотации
        type_data = os.path.splitext(annotation_file)[0].split('_')[-1]
        json_file_path = os.path.join(coco_annotations_path, annotation_file) # путь к json файлу

        # Предобработка для случая YOLOv8-obb
        preprocessing_for_yolov8_obb_model(coco_json=json_file_path)

        # Создаем папку, если ее нет
        if not autosplit:
            for folder_path in ['images', 'labels']:
                path_create=os.path.join(yolo_dataset_path, type_data.lower(), folder_path)
                os.makedirs(path_create, exist_ok=True)
            


        # открытие coco json
        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        # Получаем список изображений из JSON
        images = coco_data['images']

        # Создаем словарь с информацией о классах
        coco_categories = coco_data['categories']
        categories_dict = {category['id']-1: category['name'] for category in coco_categories}

        # Выводим информацию
        if print_info:
                print(f'Осуществляется обработка {annotation_file}')
                print(f'Имеющиеся классы: {categories_dict}')
                print('-----------------\n')

        #### Дополнительная прверка на наличие всех файлов изображений
        # Получаем список файлов изображений, для которых есть разметка в COCO
        annotated_images = set([entry['file_name'] for entry in coco_data['images']])

        # Получаем список файлов в папке с изображениями
        all_images = set(os.listdir(coco_images_path))

        # Проверяем, что все изображения из COCO размечены
        if not annotated_images.issubset(all_images):
            missing_images = annotated_images - all_images
            raise FileNotFoundError(f"Некоторые изображения, для которых есть разметка в {json_file_path}, отсутствуют в папке с изображениями. "
                                    f"Отсутствующие изображения: {missing_images}")
            
        # Проходим по изображениям и считываем аннотации
        for image in images:
            image_id = image['id']
            file_name = image['file_name']
            path_image_initial = os.path.join(coco_images_path, file_name)
            
            # Находим соответствующие аннотации для изображения
            list_of_lists_annotations = [ann['segmentation'] for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            annotations = [sublist[0] for sublist in list_of_lists_annotations]
            classes = [ann['category_id']-1 for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            
            if autosplit:
                # Генерация случайного числа от 1 до 100
                random_number = random.randint(1, 100)
                # Если случайное число <= percent_val, то type_dataset = "validation", иначе "train"
                type_dataset = "validation" if random_number <= percent_val else "train"
            else:
                type_dataset = type_data.lower()

            # Создаем объект класса ImageElement:
            element = ImageElement(
                    path_image_initial=path_image_initial,
                    path_label_initial=json_file_path,
                    img_width=image['width'],
                    img_height=image['height'],
                    image_id=image_id,
                    type_data=type_dataset,
                    path_label_final=os.path.join(yolo_dataset_path, type_dataset,
                                                'labels', os.path.splitext(file_name)[0]+'.txt'),
                    path_image_final=os.path.join(yolo_dataset_path, type_dataset,
                                                'images', file_name),
                    classes_names=[categories_dict[cl] for cl in classes],
                    classes_ids=classes,
                    point_list=annotations,
                    )
            list_of_image_elements.append(element)
            list_of_images_path.append(file_name)

            # Вывод информации об ImageElement при необходимости
            if print_info:
                print(element)

    ### Проверка на присутвие всех изображений в папке images 
    # Получаем список файлов в папке
    files_in_folder = set(os.listdir(coco_images_path))

    # Проверяем, что все файлы из списка присутствуют в папке
    missing_files = set(list_of_images_path) - files_in_folder
    extra_files = files_in_folder - set(list_of_images_path)

    # Выводим уведомление
    if missing_files:
        print(f"Отсутствующие файлы в папке {coco_images_path}: {missing_files}")

    if extra_files:
        print(f"Лишние файлы в папке {coco_images_path}: {extra_files}")


    ####### Создание конфигурации data.yaml:        
    # Создаем структуру данных для записи в data.yaml
    data_dict = {
        'names': list(categories_dict.values()),
        'nc': len(categories_dict),
        'test': 'test/images',
        'train': 'train/images',
        'val': 'validation/images'
    }
    if autosplit:
        data_dict['test'] = 'validation/images'
    
    # Путь к файлу data.yaml
    data_yaml_path = f"{yolo_dataset_path}/data.yaml"  

    # Записываем данные в файл data.yaml
    with open(data_yaml_path, 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)

    ####### Создание лейблов и копирование изображений по папкам:
    for element in list_of_image_elements:
        # Копирование изображения
        shutil.copy(element.path_image_initial, element.path_image_final)

        # Создание файла с разметкой YOLO
        with open(element.path_label_final, 'w') as yolo_label_file:
            for i in range(len(element.classes_ids)):
                class_id = element.classes_ids[i]
                class_name = element.classes_names[i]
                points = element.point_list[i]
                output_string = f'{class_id}'

                for i, point in enumerate(points):

                    if i % 2 == 0:
                        result = round(point / element.img_width, 9)
                    else:
                        result = round(point / element.img_height, 9)
                    output_string += f' {result:.6f}'
                # Запись данных в файл
                yolo_label_file.write(output_string+'\n')
                    
    print(f"Итоговая разметка в формате YOLOv8 расположена в папке - {yolo_dataset_path}.")



if __name__ == "__main__":
    main()