from pathlib import Path
from ultralytics import YOLO

def main() -> None:
    output_dir = Path("export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading/Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    print("Exporting YOLOv8n to ONNX...")
    # model.export() returns the path to the exported file
    export_path = model.export(format="onnx", imgsz=640, dynamic=False, opset=18)
    
    source_onnx = Path(export_path)
    target_onnx = output_dir / "yolov8n.onnx"
    
    if source_onnx.resolve() != target_onnx.resolve():
        if target_onnx.exists():
            target_onnx.unlink()
        source_onnx.rename(target_onnx)
        
    print(f"ONNX model saved to: {target_onnx}")

if __name__ == "__main__":
    main()
