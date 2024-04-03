from ultralytics import YOLO


def convert_model(model_file_path='model.pt'):
    print('converting model')

    YOLO(model_file_path).export(format="onnx")

    print('model converted')


if __name__ == '__main__':
    convert_model()
