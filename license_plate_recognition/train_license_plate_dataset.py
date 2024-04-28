from ultralytics import YOLO

class YOLODetectorTrainer():
    def __init__(self,
                 dataset:str='datasets/data.yaml',
                 epochs:int = 100):
        self.dataset = dataset
        self.epochs = epochs

    def train(self):
        backbone = YOLO("./yolov8n.pt")
        dataset = './datasets/data.yaml'

        results = backbone.train(data=dataset, epochs=self.epochs)

        np_model = YOLO('runs/detect/train4/weights/best.pt')

        return results, np_model