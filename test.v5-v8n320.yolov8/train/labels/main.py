from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 初始化模型
    # 如果目录下没这个文件，它会自动下载官方预训练权重
    model = YOLO('yolov8n.pt')

    # 2. 开始训练
    # data 指向 Roboflow 给你的那个 data.yaml

    model.train(
        #在字符串前面加一个 r，防止 Windows 的反斜杠报错
        data=r'C:\Users\PC\PycharmProjects\PythonProject\yolov8n_test\test.v5-v8n320.yolov8\data.yaml',
        epochs=100,      # 练 100 轮
        imgsz=320,       # 图片尺寸
        device=0,        # 有 NVIDIA 显卡就写 0，没有就写 'cpu'
        plots=True       # 自动生成训练曲线图，方便你写技术报告
    )