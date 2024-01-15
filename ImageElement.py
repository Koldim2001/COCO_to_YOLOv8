
class ImageElement:
    def __init__(
        self,
        path_image_initial: str,
        path_label_initial: str,
        img_width: int,
        img_hight: int,
        image_id: int,
        type_data: None,
        path_label_final: None,
        path_image_final: None,
        classes_names: [],
        classes_ids: [],
        point_list: [],
    ) -> None:
        self.path_image_initial = path_image_initial  # путь к исходному изображению
        self.path_label_initial = path_label_initial  # путь к исходному coco json с данными о нем

        self.img_width = img_width
        self.img_hight = img_hight

        self.image_id = image_id  # id изображения согласно coco

        self.type_data = type_data  # типа данных (train, test, valid)

        self.path_label_final = path_label_final  # путь к итоговому лейблу yolo
        self.path_image_final = path_image_final  # путь к итоговому изображению yolo формат

        # Список имен классов ex:[car,car,car,dog] - 3 объекта класса car и 1 класса dog:
        self.classes_names = classes_names
        # Список номеров классов от 0 до N-1 ex:[0,0,0,1] - 3 объекта класса 0 и 1 класса 1:
        self.classes_ids = classes_ids  

        # Список списков точек ex [[x,y,x,y,x,y],[x,y,x,yx,y,x,y]] len равен числу объектов на фото:
        self.point_list = point_list
