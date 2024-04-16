import os
import json
import shutil
import random
import click

import yaml
from shapely.geometry import Polygon
from shapely.affinity import rotate

from ImageElement import ImageElement


def preprocessing_for_yolov8_obb_model(coco_json: str, lang_ru=False):
    """
    Checks for Oriented Bounding Boxes in COCO format. If found,
    replaces the bbox and rotation of each object with the coordinates of four points in the segmentation section.
    
    Args:
    - coco_json (str): Path to the file containing COCO data in JSON format.
    - lang_ru (bool): If True, all comments will be in Russian (otherwise in English).
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
        if lang_ru:
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
    help="Folder with COCO 1.0 format dataset (can be exported from CVAT). Default is COCO_dataset",
    type=str,
)
@click.option(
    "--yolo_dataset",
    default="YOLO_dataset",
    help="Folder with the resulting YOLOv8 format dataset. Default is YOLO_dataset",
    type=str,
)
@click.option(
    "--print_info",
    default=False,
    help="Enable/Disable processing log output mode. Default is disabled",
    type=bool,
)
@click.option(
    "--autosplit",
    help="Enable/Disable automatic split into train/val. Default is disabled (uses the CVAT annotations)",
    default=False,
    type=bool,
)
@click.option(
    "--percent_val",
    help="Percentage of data for validation when using autosplit=True. Default is 25%",
    default=25,
    type=float,
)
@click.option(
    "--lang_ru",
    help="Sets the Russian language of comments, if selected value is True. English by default",
    default=False,
    type=bool,
)
def main(**kwargs):
    # ------------------ ARG parse ------------------
    coco_dataset_path = kwargs["coco_dataset"]
    yolo_dataset_path = kwargs["yolo_dataset"]
    print_info = kwargs["print_info"]
    autosplit = kwargs["autosplit"]
    percent_val = kwargs["percent_val"]
    lang_ru = kwargs["lang_ru"]

    coco_annotations_path = os.path.join(coco_dataset_path, 'annotations')
    coco_images_path = os.path.join(coco_dataset_path, 'images')

    # Check the presence of the dataset
    if not os.path.exists(coco_dataset_path):
        if lang_ru:
            raise FileNotFoundError(f"Папка с COCO датасетом '{coco_images_path}' не найдена.")
        else:
            raise FileNotFoundError(f"The COCO dataset folder '{coco_images_path}' was not found.")

    # Check the presence of the images folder
    if not os.path.exists(coco_images_path):
        if lang_ru:
            raise FileNotFoundError(f"Папка с изображениями '{coco_images_path}' не найдена. "
                            f"Убедитесь, что вы загрузили разметку COCO так, чтобы имелась папка со всеми изображениями.")
        else:
            raise FileNotFoundError(f"The images folder '{coco_images_path}' was not found. "
                            f"Make sure you have uploaded COCO annotations so that there is a folder with all images.")

    # Check if the annotations folder exists
    if not os.path.exists(coco_annotations_path):
        if lang_ru:
            raise FileNotFoundError(f"The folder with json files '{coco_annotations_path}' was not found.")
        else:
            raise FileNotFoundError(f"Папка с json файлами '{coco_annotations_path}' не найдена.")

    list_of_image_elements = []
    list_of_images_path = []

    # Get a list of all files in the annotations folder
    annotation_files = os.listdir(coco_annotations_path)

    shutil.rmtree(yolo_dataset_path, ignore_errors=True) # Clear old data in the folder

    if autosplit:
        for folder_path in ['images', 'labels']:
            for type in ['validation', 'train']:
                path_create=os.path.join(yolo_dataset_path, type, folder_path)
                os.makedirs(path_create, exist_ok=True)

    ### Check for duplicates in different subsets ###
    # Create a dictionary to store files and their corresponding JSON files
    file_json_mapping = {}

    # Iterate through annotation files
    for annotation_file in annotation_files:
        json_file_path = os.path.join(coco_annotations_path, annotation_file)
        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        # Get the list of images from JSON
        images = coco_data['images']

        # Iterate through images and update the file_json_mapping dictionary
        for image in images:
            file_name = image['file_name']
            if file_name not in file_json_mapping:
                file_json_mapping[file_name] = [annotation_file]
            else:
                file_json_mapping[file_name].append(annotation_file)

    # Check if any file has more than one occurrence
    for file_name, json_files in file_json_mapping.items():
        if len(json_files) > 1:
            if lang_ru:
                print(f"Файл {file_name} встречается в следующих JSON файлах: {json_files}")
                print(f'В каком-либо из JSON файлов удалите в разделе "images" словарь ' \
                      f'с описанием этой фотографии, иначе будет ошибка при выполнении кода')
                raise SystemExit
            else:
                print(f"The file {file_name} appears in the following JSON files: {json_files}")
                print(f"Remove the dictionary describing this photo from the 'images' section in " \
                      f"one of the JSON files, otherwise there will be an error when running the code.")
                raise SystemExit

    ### Run the main code: ###
           
    # Iterate through annotation files
    for annotation_file in annotation_files:
        # Parse the image file name from the annotation file
        type_data = os.path.splitext(annotation_file)[0].split('_')[-1]
        json_file_path = os.path.join(coco_annotations_path, annotation_file) # path to the json file

        # Preprocessing for YOLOv8-obb
        preprocessing_for_yolov8_obb_model(coco_json=json_file_path, lang_ru=lang_ru)

        # Create folder if it doesn't exist
        if not autosplit:
            for folder_path in ['images', 'labels']:
                path_create=os.path.join(yolo_dataset_path, type_data.lower(), folder_path)
                os.makedirs(path_create, exist_ok=True)

        # Open coco json
        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        # Get the list of images from JSON
        images = coco_data['images']

        # Create a dictionary with class information
        coco_categories = coco_data['categories']
        categories_dict = {category['id']-1: category['name'] for category in coco_categories}

        # Print information
        if print_info:
            if lang_ru:
                print(f'Осуществляется обработка {annotation_file}')
                print(f'Имеющиеся классы: {categories_dict}')
            else:
                print(f'Processing {annotation_file}')
                print(f'Available classes: {categories_dict}')
            print('-----------------\n')

        #### Additional check for the presence of all image files
        # Get the list of image files with annotations in COCO
        annotated_images = set([entry['file_name'] for entry in coco_data['images']])

        # Get the list of files in the images folder
        all_images = set(os.listdir(coco_images_path))

        # Check that all images from COCO are annotated
        if not annotated_images.issubset(all_images):
            missing_images = annotated_images - all_images
            if lang_ru:
                raise FileNotFoundError(f"Некоторые изображения, для которых есть разметка в {json_file_path}, отсутствуют в папке с изображениями. "
                                    f"Отсутствующие изображения: {missing_images}")
            else:
                raise FileNotFoundError(f"Some images annotated in {json_file_path} are missing from the images folder. "
                                    f"Missing images: {missing_images}")
                

        # Iterate through images and read annotations
        for image in images:
            image_id = image['id']
            file_name = image['file_name']
            path_image_initial = os.path.join(coco_images_path, file_name)
            
            # Find corresponding annotations for the image
            list_of_lists_annotations = [ann['segmentation'] for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            try:
                annotations = [sublist[0] for sublist in list_of_lists_annotations]
            except:
                if lang_ru:
                    print(f"В разметке фотографии {file_name} имеются объекты, не являющиеся полигонами. "\
                        f"\nНеобходимо, чтобы все объекты для обучения YOLOv8-seg были размечены как полигоны! "\
                        f"\nИсправьте это и заново выгрузите датасет.")
                else:
                    print(f"The annotations for the image {file_name} contain objects that are not polygons. "\
                      f"\nAll objects for training YOLOv8-seg must be annotated as polygons! "\
                      f"\nPlease correct this and reload the dataset.")
                raise SystemExit
            
            classes = [ann['category_id']-1 for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            
            if autosplit:
                # Generate a random number from 1 to 100
                random_number = random.randint(1, 100)
                # If the random number <= percent_val, then type_dataset = "validation", otherwise "train"
                type_dataset = "validation" if random_number <= percent_val else "train"
            else:
                type_dataset = type_data.lower()

            # Create an instance of the ImageElement class:
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

            # Print information about ImageElement if necessary
            if print_info:
                print(element)

    ### Check for the presence of all images in the images folder 
    # Get the list of files in the folder
    files_in_folder = set(os.listdir(coco_images_path))

    # Check that all files from the list are present in the folder
    missing_files = set(list_of_images_path) - files_in_folder
    extra_files = files_in_folder - set(list_of_images_path)

    # Display notification
    if missing_files:
        if lang_ru:
            print(f"Отсутствующие файлы в папке {coco_images_path}: {missing_files}")
        else:
            print(f"Missing files in the folder {coco_images_path}: {missing_files}")

    if extra_files:
        if lang_ru:
            print(f"Лишние файлы в папке {coco_images_path}: {extra_files}")
        else:
            print(f"Extra files in the folder {coco_images_path}: {extra_files}")

    # Creating data.yaml configuration:
    # Create a data structure for writing to data.yaml
    data_dict = {
        'names': list(categories_dict.values()),
        'nc': len(categories_dict),
        'test': 'test/images',
        'train': 'train/images',
        'val': 'validation/images'
    }
    if autosplit:
        data_dict['test'] = 'validation/images'

    # Path to the data.yaml file
    data_yaml_path = f"{yolo_dataset_path}/data.yaml"  

    # Write data to the data.yaml file
    with open(data_yaml_path, 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)

    # Creating labels and copying images to folders:
    for element in list_of_image_elements:
        # Copying the image
        shutil.copy(element.path_image_initial, element.path_image_final)

        # Creating a YOLO annotation file
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
                # Writing data to the file
                yolo_label_file.write(output_string+'\n')
                    
    if lang_ru:
        print(f"Итоговая разметка в формате YOLOv8 расположена в папке - {yolo_dataset_path}.")
    else:
        print(f"The final YOLOv8 format annotations are located in the folder - {yolo_dataset_path}.")                  



if __name__ == "__main__":
    main()