from os import environ
from shutil import move

from ultralytics import settings, YOLO


def train_model(
        data_folder='./datasets', batch_size=1, epochs=1):
    print('training model')

    batch_size = batch_size or int(environ.get('batch_size', 4))
    epochs = epochs or int(environ.get('epochs', 2))

    settings.update({'datasets_dir': data_folder})
    model = YOLO('yolov8m.pt')

    results = model.train(
        data=f'{data_folder}/data.yaml',
        epochs=epochs,
        imgsz=640,
        batch=batch_size
    )
    print(f'training run complete\nresults: {results}')

    move('./runs/detect/train/weights/best.pt', 'model.pt')

    print('model training done')


if __name__ == '__main__':
    train_model(data_folder='/datasets')
