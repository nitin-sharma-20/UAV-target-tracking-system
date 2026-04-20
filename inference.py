import cv2
import onnxruntime as ort
import numpy as np
import time

def main():
    # File paths
    model_path = "best.onnx"
    video_path = "test_video.mp4" 
    
    print("Loading ONNX model for CPU inference...")
    # Force CPU execution to simulate low-power edge hardware
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Get model input requirements dynamically
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_w, input_h = input_shape[3], input_shape[2] 
    
    # Open the local video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}. Verify the file exists in this directory.")
        return

    print("Starting edge simulation... Press 'q' to quit.")
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        orig_h, orig_w = frame.shape[:2]

        # Preprocess the frame for the ONNX model
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        # Execute CPU inference
        outputs = session.run(None, {input_name: img})
        predictions = np.squeeze(outputs[0]).T 

        boxes = []
        scores = []
        class_ids = []

        x_scale = orig_w / input_w
        y_scale = orig_h / input_h

        # Extract bounding boxes and confidence scores
        for row in predictions:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]
            
            # Filter low-confidence detections
            if score > 0.5: 
                x_center, y_center, w, h = row[0], row[1], row[2], row[3]
                
                # Scale coordinates back to original video resolution
                left = int((x_center - w / 2) * x_scale)
                top = int((y_center - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                
                boxes.append([left, top, width, height])
                scores.append(float(score))
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)

        # Draw final bounding boxes on the frame
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {class_ids[i]} Conf: {scores[i]:.2f}", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate CPU FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"CPU FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the video feed
        cv2.imshow("UAV Edge Tracking Simulation", frame)

        # Check for user exit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()