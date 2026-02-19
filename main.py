from __future__ import annotations

import os
import sys
import threading
from random import randint

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from torch.cuda import is_available


def log_error(message: str, exc: Exception | None = None) -> None:
    """Print non-fatal errors in a consistent way."""
    print(f"[ERROR] {message}")
    if exc is not None:
        print(f"        {type(exc).__name__}: {exc}")


facenet_model = InceptionResnetV1(pretrained="vggface2").eval()
device = "cuda" if is_available() else "cpu"
mtcnn = MTCNN(device=device)


def load_settings_from_file(path: str = "educational.txt") -> list[dict]:
    """
    Parse educational.txt and return a list of setting dicts.
    Each line is: name|default|description
    """
    settings = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 2)
                if len(parts) == 3:
                    settings.append({
                        "name": parts[0],
                        "default": parts[1],
                        "description": parts[2],
                    })
    except Exception as e:
        log_error("Failed to load educational.txt", e)
    return settings


def save_face(face_image, match_file):
    """Save a detected face image to disk. Never raises; logs on failure."""
    try:
        face_image = Image.fromarray(face_image)

        if not match_file:
            match_file = "unknown"

        os.makedirs("./saved_faces", exist_ok=True)

        filename = f"./saved_faces/{match_file}_{randint(1, 10000)}.jpg"
        face_image.save(filename)

        return filename
    except Exception as e:
        log_error("Failed to save face image", e)
        return None


def save_unrecognized_face_and_add_embedding(
    face_image, face_embedding, embeddings: dict, live_dir: str = "./live_detected"
):
    """
    Save an unrecognized face into live_detected as person[n+1].jpg and
    immediately add its embedding into the in-memory embeddings dict so that
    the same person is recognized in subsequent frames.
    """
    try:
        os.makedirs(live_dir, exist_ok=True)

        allowed_exts = (".jpg", ".jpeg", ".png")
        existing_files = [
            f
            for f in os.listdir(live_dir)
            if os.path.isfile(os.path.join(live_dir, f))
            and f.lower().endswith(allowed_exts)
        ]
        next_index = len(existing_files) + 1

        filename = os.path.join(live_dir, f"person{next_index}.jpg")

        face_image = Image.fromarray(face_image)
        face_image.save(filename)

        if face_embedding is not None:
            try:
                face_embedding_tuple = tuple(face_embedding.tolist())
                embeddings[face_embedding_tuple] = filename
            except Exception as e:
                log_error(
                    "Failed to add live embedding for unrecognized face", e
                )

        return filename
    except Exception as e:
        log_error("Failed to save unrecognized face to live_detected", e)
        return None


def is_face_high_quality_for_live_detect(
    face_image,
    face_area: int,
    min_live_area: int,
    sharpness_threshold: float = 100.0,
):
    """
    Return True if this face crop is good enough to be stored in live_detected.
    """
    try:
        if face_area < min_live_area:
            return False

        if face_image is None or face_image.size == 0:
            return False

        h, w = face_image.shape[:2]
        if h == 0 or w == 0:
            return False

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.6 or aspect_ratio > 1.8:
            return False

        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < sharpness_threshold:
            return False

        return True
    except Exception as e:
        log_error("Error during face quality evaluation", e)
        return False


def get_embedding(face):
    """Return embedding for a single face crop. Returns None on any failure."""
    try:
        tensor_image = mtcnn(face)
    except Exception as e:
        log_error("Error during face detection in get_embedding", e)
        return None

    if tensor_image is None:
        return None

    try:
        return facenet_model(tensor_image.unsqueeze(0)).detach().numpy()[0]
    except Exception as e:
        log_error("Error during embedding computation", e)
        return None


def face_matching(face_embedding, embeddings: list | dict, similarity_threshold: float):
    """
    Matches the face_embedding with the embeddings and returns match info
    or False.
    """
    for i, embedding in enumerate(embeddings):
        cosine_similarity = cosine(face_embedding, embedding)

        if cosine_similarity < similarity_threshold:
            info = (cosine_similarity * 100, i + 1, embeddings[embedding])
            return info

    return False


def load_embeddings(load_amount: int, images_path: str) -> dict:
    """
    Loads the face embeddings from the images in the images_path folder.
    """
    try:
        if images_path[-1] != "/":
            images_path += "/"
    except Exception as e:
        log_error("Invalid images_path provided", e)
        return {}

    if not os.path.exists(images_path):
        log_error("Images path does not exist")
        return {}

    allowed_image_extensions = ["jpg", "jpeg", "png"]

    filtered_files = [
        file
        for file in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, file))
        and any(file.endswith(ext) for ext in allowed_image_extensions)
    ]

    images_embeddings: dict = {}
    for i, file in enumerate(filtered_files, start=1):
        if i > load_amount:
            break

        try:
            face = Image.open(images_path + file).convert("RGB")
        except Exception as e:
            log_error(f"Failed to open image {file}", e)
            continue

        face_embedding = get_embedding(face)

        if face_embedding is None:
            log_error(f"Face not detected or embedding failed in image {file}")
            continue

        try:
            face_embedding_tuple = tuple(face_embedding.tolist())
            images_embeddings[face_embedding_tuple] = file
        except Exception as e:
            log_error(f"Failed to store embedding for image {file}", e)

    if not images_embeddings:
        log_error("No valid face embeddings were loaded")

    return images_embeddings


# ---------------------------------------------------------------------------
# Video processing thread
# ---------------------------------------------------------------------------

class VideoThread(QThread):
    """Runs video capture and face recognition in a separate thread."""

    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, video_source, embeddings, settings, lock, parent=None):
        super().__init__(parent)
        self.video_source = video_source
        self.embeddings = embeddings
        self.settings = settings
        self.lock = lock
        self._running = True

    def run(self):
        video_capture = cv2.VideoCapture(self.video_source)
        if not video_capture.isOpened():
            log_error(f"Unable to open video source: {self.video_source}")
            return

        frame_count = 0
        while self._running:
            try:
                ret, frame = video_capture.read()
                frame_count += 1
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                min_probability = self.settings.get("min_probability", 0.95)
                max_distance = self.settings.get("max_distance", 0.4)
                min_live_area = self.settings.get("min_live_area", 4900)

                try:
                    boxes, probs = mtcnn.detect(rgb_frame)
                except Exception as e:
                    log_error(
                        f"Error during face detection on frame {frame_count}", e
                    )
                    boxes, probs = None, None

                if boxes is not None:
                    for box, prob in zip(boxes, probs):
                        if prob < min_probability:
                            continue
                        x1, y1, x2, y2 = box.astype(int)
                        face = rgb_frame[y1:y2, x1:x2]
                        face_area = max(0, x2 - x1) * max(0, y2 - y1)
                        face_embedding = get_embedding(face)

                        if face_embedding is None:
                            continue

                        with self.lock:
                            match_info = (
                                face_matching(
                                    face_embedding, self.embeddings, max_distance
                                )
                                if self.embeddings
                                else None
                            )

                        if match_info:
                            cosine_similarity, embedding_index, match_file = (
                                match_info
                            )
                            cv2.rectangle(
                                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                            )
                        else:
                            cv2.rectangle(
                                frame, (x1, y1), (x2, y2), (0, 0, 255), 2
                            )
                            if is_face_high_quality_for_live_detect(
                                face, face_area, min_live_area
                            ):
                                with self.lock:
                                    save_unrecognized_face_and_add_embedding(
                                        face, face_embedding, self.embeddings
                                    )
                        if match_info:
                            save_face(face, match_file)

                # Convert BGR frame to RGB for Qt display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(display_frame)

            except Exception as e:
                log_error(
                    f"Unexpected error in main loop at frame {frame_count}", e
                )
                continue

        video_capture.release()

    def stop(self):
        self._running = False
        self.wait()


# ---------------------------------------------------------------------------
# Slideshow widget
# ---------------------------------------------------------------------------

SLIDESHOW_SLIDES = [
    {
        "title": "Welcome to Face Recognizer",
        "body": (
            "This application uses deep learning to detect and recognize faces "
            "in real-time video streams. Press Start on the settings panel to begin."
        ),
    },
    {
        "title": "How It Works",
        "body": (
            "Faces are detected using MTCNN and encoded with a FaceNet model. "
            "Each detected face is compared against a database of known embeddings "
            "to find the closest match."
        ),
    },
    {
        "title": "Live Detection",
        "body": (
            "Unrecognized faces that meet quality thresholds are automatically saved "
            "to the live_detected folder so they can be recognized in future frames."
        ),
    },
]


class SlideshowWidget(QWidget):
    """A simple slideshow displaying a title, image placeholder, and paragraph."""

    def __init__(self, slides: list[dict] | None = None, parent=None):
        super().__init__(parent)
        self.slides = slides or SLIDESHOW_SLIDES
        self._current = 0
        self._build_ui()
        self._show_slide(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Title
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Image placeholder (a bordered box)
        self.image_label = QLabel()
        self.image_label.setFixedSize(260, 180)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setStyleSheet(
            "background-color: #e0e0e0; color: #888; font-size: 13px; border: 2px solid #aaa;"
        )
        self.image_label.setText("[ Image Placeholder ]")
        layout.addWidget(self.image_label, alignment=Qt.AlignHCenter)

        # Body text
        self.body_label = QLabel()
        self.body_label.setWordWrap(True)
        self.body_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.body_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.body_label)

        # Push arrows to the bottom-right
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        arrow_layout = QHBoxLayout()
        arrow_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.prev_btn = QPushButton("\u25C0")
        self.prev_btn.setFixedSize(36, 36)
        self.prev_btn.clicked.connect(self._prev_slide)
        arrow_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("\u25B6")
        self.next_btn.setFixedSize(36, 36)
        self.next_btn.clicked.connect(self._next_slide)
        arrow_layout.addWidget(self.next_btn)

        layout.addLayout(arrow_layout)

    def _show_slide(self, index: int):
        if not self.slides:
            return
        self._current = index % len(self.slides)
        slide = self.slides[self._current]
        self.title_label.setText(slide.get("title", ""))
        self.body_label.setText(slide.get("body", ""))

    def _prev_slide(self):
        self._show_slide(self._current - 1)

    def _next_slide(self):
        self._show_slide(self._current + 1)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """PyQt5 main window with video stream and live-adjustable settings."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognizer")

        self.video_thread: VideoThread | None = None
        self.embeddings: dict = {}
        self.embeddings_lock = threading.Lock()
        self.settings: dict = {}
        self._settings_data = load_settings_from_file()

        self._build_ui()
        self._load_defaults()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: video display
        self.video_label = QLabel("Press Start to begin")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, stretch=3)

        # Right: settings panel
        settings_group = QGroupBox("Settings")
        form = QFormLayout()

        self.images_path_edit = QLineEdit()
        form.addRow("images_path:", self.images_path_edit)

        self.load_amount_spin = QSpinBox()
        self.load_amount_spin.setRange(1, 1000)
        form.addRow("load_amount:", self.load_amount_spin)

        self.video_path_edit = QLineEdit()
        form.addRow("video_path:", self.video_path_edit)

        self.use_webcam_check = QCheckBox()
        form.addRow("use_webcam:", self.use_webcam_check)

        self.min_probability_spin = QDoubleSpinBox()
        self.min_probability_spin.setRange(0.0, 1.0)
        self.min_probability_spin.setSingleStep(0.05)
        self.min_probability_spin.setDecimals(2)
        form.addRow("min_probability:", self.min_probability_spin)

        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(0.0, 1.0)
        self.max_distance_spin.setSingleStep(0.05)
        self.max_distance_spin.setDecimals(2)
        form.addRow("max_distance:", self.max_distance_spin)

        self.min_live_area_spin = QSpinBox()
        self.min_live_area_spin.setRange(0, 1000000)
        form.addRow("min_live_area:", self.min_live_area_spin)

        # Description labels loaded from educational.txt
        for s in self._settings_data:
            desc_label = QLabel(s["description"])
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: gray; font-size: 10px;")
            form.addRow("", desc_label)

        settings_group.setLayout(form)

        right_panel = QVBoxLayout()
        right_panel.addWidget(settings_group)

        # Connect live-update signals for numeric settings
        self.min_probability_spin.valueChanged.connect(self._apply_live_settings)
        self.max_distance_spin.valueChanged.connect(self._apply_live_settings)
        self.min_live_area_spin.valueChanged.connect(self._apply_live_settings)

        # Start / Stop buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        right_panel.addLayout(btn_layout)
        layout.addLayout(right_panel, stretch=1)

        # Far-right: slideshow panel
        self.slideshow = SlideshowWidget()
        layout.addWidget(self.slideshow, stretch=1)

    def _load_defaults(self):
        """Populate widgets with defaults from educational.txt."""
        defaults = {s["name"]: s["default"] for s in self._settings_data}

        self.images_path_edit.setText(defaults.get("images_path", "./sample_images/"))
        self.load_amount_spin.setValue(int(defaults.get("load_amount", "10")))
        self.video_path_edit.setText(
            defaults.get("video_path", "./faceexamplevideo.mkv")
        )
        self.use_webcam_check.setChecked(defaults.get("use_webcam", "False") == "True")
        self.min_probability_spin.setValue(
            float(defaults.get("min_probability", "0.95"))
        )
        self.max_distance_spin.setValue(float(defaults.get("max_distance", "0.4")))
        self.min_live_area_spin.setValue(int(defaults.get("min_live_area", "4900")))

    # -- Settings helpers ---------------------------------------------------

    def _read_settings(self) -> dict:
        return {
            "images_path": self.images_path_edit.text(),
            "load_amount": self.load_amount_spin.value(),
            "video_path": self.video_path_edit.text(),
            "use_webcam": self.use_webcam_check.isChecked(),
            "min_probability": self.min_probability_spin.value(),
            "max_distance": self.max_distance_spin.value(),
            "min_live_area": self.min_live_area_spin.value(),
        }

    def _apply_live_settings(self):
        """Push changed numeric settings to the running thread."""
        self.settings["min_probability"] = self.min_probability_spin.value()
        self.settings["max_distance"] = self.max_distance_spin.value()
        self.settings["min_live_area"] = self.min_live_area_spin.value()

    # -- Start / Stop -------------------------------------------------------

    def _on_start(self):
        if self.video_thread and self.video_thread.isRunning():
            return

        self.settings = self._read_settings()

        images_path = self.settings["images_path"]
        load_amount = self.settings["load_amount"]

        try:
            self.embeddings = load_embeddings(load_amount, images_path)
        except Exception as e:
            log_error("Failed to load embeddings", e)
            self.embeddings = {}

        if self.settings["use_webcam"]:
            video_source = 0
        else:
            video_source = self.settings["video_path"]
            allowed_video_extensions = [".mkv", ".mp4", ".avi", ".mov", ".wmv"]
            if not os.path.exists(video_source):
                log_error("Video file does not exist")
                return
            if not any(video_source.endswith(ext) for ext in allowed_video_extensions):
                log_error("Video file extension not supported")
                return

        self.video_thread = VideoThread(
            video_source, self.embeddings, self.settings, self.embeddings_lock
        )
        self.video_thread.frame_ready.connect(self._update_frame)
        self.video_thread.finished.connect(self._on_thread_finished)
        self.video_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_stop(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self.video_label.setText("Stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_thread_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Stream ended")

    # -- Frame display ------------------------------------------------------

    @pyqtSlot(np.ndarray)
    def _update_frame(self, rgb_frame: np.ndarray):
        """Convert a numpy RGB frame to QPixmap and display it."""
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    # -- Cleanup ------------------------------------------------------------

    def closeEvent(self, event):
        self._on_stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 700)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.")
    except Exception as e:
        log_error("Unhandled error in application", e)
