
class ImageElement:
    def __init__(
        self,
        path_image_initial: str,
        path_label_initial: str,
        img_width: int,
        img_height: int,
        image_id: int,
        type_data: str,
        path_label_final: str,
        path_image_final: str,
        classes_names: list,
        classes_ids: list,
        point_list: list,
    ) -> None:
        self.path_image_initial = path_image_initial  # path to the original image
        self.path_label_initial = path_label_initial  # path to the original COCO json with its data

        self.img_width = img_width
        self.img_height = img_height

        self.image_id = image_id  # image id according to COCO

        self.type_data = type_data  # type of data (train, test, valid)

        self.path_label_final = path_label_final  # path to the final YOLO label
        self.path_image_final = path_image_final  # path to the final YOLO image format

        # List of class names ex: [car, car, car, dog] - 3 objects of class car and 1 object of class dog:
        self.classes_names = classes_names
        # List of class numbers from 0 to N-1 ex: [0, 0, 0, 1] - 3 objects of class 0 and 1 object of class 1:
        self.classes_ids = classes_ids  

        # List of lists of points ex [[x, y, x, y, x, y], [x, y, x, y, x, y]] length equals the number of objects in the photo:
        self.point_list = point_list
        
    def __str__(self):
        # Converting each segmentation to the number of points
        segmentations_lengths = [len(segmentation) // 2 for segmentation in self.point_list]
        return (
            f"ImageElement info:\n"
            f" - path_image_initial: {self.path_image_initial}\n"
            f" - path_label_initial: {self.path_label_initial}\n"
            f" - img_width: {self.img_width}\n"
            f" - img_height: {self.img_height}\n"
            f" - image_id: {self.image_id}\n"
            f" - type_data: {self.type_data}\n"
            f" - path_label_final: {self.path_label_final}\n"
            f" - path_image_final: {self.path_image_final}\n"
            f" - classes_names: {self.classes_names}\n"
            f" - classes_ids: {self.classes_ids}\n"
            f" - points_amount: {segmentations_lengths}\n"
        )
