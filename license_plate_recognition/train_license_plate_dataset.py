from ultralytics import YOLO
import os
import requests
import zipfile

''' You should get your own url(key) from roboflow (https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
    and use that to train yolov8n with that dataset.
    if you have your own dataset feel free to use that instead of using this dataset to train your model.
'''
class YOLODetectorTrainer():
    def __init__(self,
                 url: str=None,
                 dataset:str='datasets/data.yaml',
                 epochs:int = 100):
        self.dataset = dataset
        self.epochs = epochs
        self.url = url

    def train(self):
        backbone = YOLO("./yolov8n.pt")
        if os.path.exists(self.dataset):
            dataset = './datasets/data.yaml'
        else :
            extract_dir = './datasets'
            os.makedirs(extract_dir, exist_ok=True)
            response = requests.get(self.url)
            zip_file_path = os.path.join(extract_dir, "data.zip")

            with open(zip_file_path, "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        results = backbone.train(data=dataset, epochs=self.epochs)

        np_model = YOLO('runs/detect/train4/weights/best.pt')

        return results, np_model