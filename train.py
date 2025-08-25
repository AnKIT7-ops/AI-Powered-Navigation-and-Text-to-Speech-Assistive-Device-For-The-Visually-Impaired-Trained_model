from ultralytics import YOLO

def main():
    # Load YOLOv8n (nano) pre-trained weights
    model = YOLO("yolov8n.pt")

    model.train(
        data="coco.yaml",        # dataset YAML
        epochs=50,               # start with 50 (can extend later)
        imgsz=640,               # standard COCO size
        batch=32,                # sweet spot for RTX 4050 (6–8GB VRAM)
        workers=4,               # dataloader workers
        device=0,                # use GPU (RTX 4050)
        optimizer="SGD",         # SGD is stable, AdamW can be faster but more VRAM
        cos_lr=True,             # cosine learning rate scheduler (faster convergence)
        cache=True,              # cache images in RAM → speeds up training
        amp=True,                # automatic mixed precision → faster
        patience=20,             # early stopping if no improvement
        project="runs/train",
        name="yolov8n_coco_4050"
    )

if __name__ == "__main__":
    main()
