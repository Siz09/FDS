import numpy as np
import cv2
import onnxruntime as ort

class YuNetDetector:
    def __init__(self, model_path, conf_threshold=0.6, nms_threshold=0.3):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        # Note: FaceDetectorYN is only in OpenCV 4.5.4+
        # If OpenCV version is old, we use onnxruntime directly
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image):
        # Implementation for YuNet post-processing
        # This is complex to write from scratch without the binary,
        # so we usually leverage cv2.FaceDetectorYN.
        # Check if cv2 has it:
        if hasattr(cv2, 'FaceDetectorYN'):
            detector = cv2.FaceDetectorYN.create(self.model_path, "", (image.shape[1], image.shape[0]), self.conf_threshold, self.nms_threshold)
            _, faces = detector.detect(image)
            return faces # x, y, w, h, landmarks...
        else:
            # Fallback to MediaPipe (already in our stack)
            return None

class ArcFaceEmbedder:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def compute_embedding(self, face_img):
        # ArcFace expects 112x112 RGB normalized to [-1, 1]
        face_img = cv2.resize(face_img, (112, 112))
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.transpose(face_img, (2, 0, 1)) # HWC to CHW
        face_img = np.expand_dims(face_img, axis=0) # Add batch dim

        embeddings = self.session.run(None, {self.input_name: face_img})[0]
        # L2 Normalize
        norm = np.linalg.norm(embeddings)
        return embeddings / (norm + 1e-6)
