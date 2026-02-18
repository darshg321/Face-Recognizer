from __future__ import annotations

import os
import tkinter as tk
from random import randint

import cv2
from PIL import Image
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


def settings_gui(settings: list[dict]) -> dict | None:
    """
    Display a minimal tkinter GUI with the settings from educational.txt.
    Returns a dict of {name: value} when the user clicks Start,
    or None if the window is closed.
    """
    result = {}
    root = tk.Tk()
    root.title("Face Recognizer Settings")

    entries = {}
    for i, setting in enumerate(settings):
        label = tk.Label(root, text=f"{setting['name']}:", anchor="w")
        label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

        if setting["default"] in ("True", "False"):
            var = tk.BooleanVar(value=setting["default"] == "True")
            widget = tk.Checkbutton(root, variable=var)
            widget.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            entries[setting["name"]] = var
        else:
            entry = tk.Entry(root, width=40)
            entry.insert(0, setting["default"])
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            entries[setting["name"]] = entry

        desc = tk.Label(root, text=setting["description"], fg="gray",
                        wraplength=300, anchor="w", justify="left")
        desc.grid(row=i, column=2, sticky="w", padx=5, pady=2)

    def on_start():
        for name, widget in entries.items():
            if isinstance(widget, tk.BooleanVar):
                result[name] = widget.get()
            else:
                result[name] = widget.get()
        root.destroy()

    start_btn = tk.Button(root, text="Start", command=on_start)
    start_btn.grid(row=len(settings), column=1, pady=10)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    return result if result else None


def get_settings() -> tuple:
    """
    Load settings from educational.txt, show the GUI, and return
    the validated settings tuple.
    """
    settings = load_settings_from_file()
    if not settings:
        raise Exception("No settings found in educational.txt")

    values = settings_gui(settings)
    if values is None:
        raise Exception("Settings window was closed without starting")

    images_path = values.get("images_path", "./sample_images/")
    images_to_load = int(values.get("load_amount", "10"))
    video_path = values.get("video_path", "./faceexamplevideo.mkv")
    use_webcam = values.get("use_webcam", False)
    min_probability = float(values.get("min_probability", "0.95"))
    max_distance = float(values.get("max_distance", "0.4"))
    min_live_area = int(values.get("min_live_area", "4900"))

    if use_webcam:
        video_source = 0
    else:
        allowed_video_extensions = [".mkv", ".mp4", ".avi", ".mov", ".wmv"]
        if not os.path.exists(video_path):
            raise Exception("Video file does not exist")
        if not any(video_path.endswith(ext) for ext in allowed_video_extensions):
            raise Exception("Video file extension not supported")
        video_source = video_path

    return (
        images_to_load,
        images_path,
        video_source,
        min_probability,
        max_distance,
        use_webcam,
        min_live_area,
    )

def save_face(face_image, match_file):
    """Save a detected face image to disk. Never raises; logs on failure."""
    try:
        # face_image needs to be array or PIL image
        face_image = Image.fromarray(face_image)

        if not match_file:
            match_file = "unknown"

        # ensure output directory exists
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
        # ensure output directory exists
        os.makedirs(live_dir, exist_ok=True)

        # count existing image files to determine the next index
        allowed_exts = (".jpg", ".jpeg", ".png")
        existing_files = [
            f
            for f in os.listdir(live_dir)
            if os.path.isfile(os.path.join(live_dir, f))
            and f.lower().endswith(allowed_exts)
        ]
        next_index = len(existing_files) + 1

        filename = os.path.join(live_dir, f"person{next_index}.jpg")

        # save the face image
        face_image = Image.fromarray(face_image)
        face_image.save(filename)

        # add this new face to embeddings so it becomes recognized next time
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
    Heuristics:
    - must be at least min_live_area pixels in area
    - must be reasonably sharp (Laplacian variance above threshold)
    - must have a non-extreme aspect ratio (rough proxy for straight-ish angle)
    """
    try:
        # area check
        if face_area < min_live_area:
            return False

        if face_image is None or face_image.size == 0:
            return False

        h, w = face_image.shape[:2]
        if h == 0 or w == 0:
            return False

        # very wide or very tall crops are often partial/profile faces
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.6 or aspect_ratio > 1.8:
            return False

        # blur / sharpness check using variance of Laplacian
        # (larger variance => sharper image)
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
    # face is an RGB frame
    try:
        tensor_image = mtcnn(face)
    except Exception as e:
        # This gracefully handles internal MTCNN errors like empty box lists.
        log_error("Error during face detection in get_embedding", e)
        return None

    if tensor_image is None:
        # No face detected in this crop
        return None

    try:
        return facenet_model(tensor_image.unsqueeze(0)).detach().numpy()[0]
    except Exception as e:
        log_error("Error during embedding computation", e)
        return None

def face_matching(face_embedding, embeddings: list | dict, similarity_threshold: float):
    """
    Matches the face_embedding with the embeddings in the embeddings
    and returns True if a match is found.

    :param face_embedding: The MTCNN embedding of the face to match
    :param embeddings: A dict of MTCNN embeddings of the faces to match with.
                       The key has to be the embedding tuple, and the value the file name.
    :param similarity_threshold: The threshold to match the face with
    :return: A tuple containing the cosine similarity in percent, the index of the
             embedding in the dict iteration and the name of the file if a match is found,
             False otherwise.
    """
    for i, embedding in enumerate(embeddings):
        cosine_similarity = cosine(face_embedding, embedding)

        if cosine_similarity < similarity_threshold:
            info = (cosine_similarity * 100, i + 1, embeddings[embedding])
            return info

    return False

def load_embeddings(load_amount: int, images_path: str) -> dict:
    """
    Loads the face embeddings from the images in the images_path folder
    and returns a dict of the MTCNN embeddings, key being a tuple of the embeddings,
    and value being the path of the image that corresponds.
    If there's no face detected in the image, it will raise an exception.
    Supported formats: .jpg, .jpeg, .png

    :param load_amount: The amount of images to load after filtering out unsupported formats
    :param images_path: The path to the images folder
    :return: A dict of the MTCNN embeddings, key is a tuple of the embedding, value is the image path
    NOTE: This function logs problems instead of raising, so that the app
    can continue running even if some images are bad.
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

    # filter the files, removing folders and extensions that aren't allowed
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

        # load the image
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

def main():
    try:
        (
            images_to_load,
            images_path,
            video_source,
            min_probability,
            max_distance,
            use_webcam,
            min_live_area,
        ) = get_settings()
    except Exception as e:
        log_error("Failed to get settings", e)
        return

    try:
        embeddings = load_embeddings(images_to_load, images_path)
    except Exception as e:
        log_error("Failed to load embeddings", e)
        embeddings = {}

    if not embeddings:
        print(
            "Warning: No embeddings loaded; face matching will be disabled. "
            "Faces will still be detected and saved."
        )

    # video_source is either a path or a webcam index (0)
    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        log_error(f"Unable to open video source: {video_source}")
        return
    frame_count = 0

    while True:
        try:
            ret, frame = video_capture.read()
            frame_count += 1
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
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
                    # MTCNN returns [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box.astype(int)
                    face = rgb_frame[y1:y2, x1:x2]
                    face_area = max(0, x2 - x1) * max(0, y2 - y1)
                    face_embedding = get_embedding(face)

                    if face_embedding is None:
                        continue

                    match_info = (
                        face_matching(face_embedding, embeddings, max_distance)
                        if embeddings
                        else None
                    )

                    if match_info:
                        cosine_similarity, embedding_index, match_file = match_info
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
                            save_unrecognized_face_and_add_embedding(
                                face, face_embedding, embeddings
                            )
                    if match_info:
                        save_face(face, match_file)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            log_error(
                f"Unexpected error in main loop at frame {frame_count}", e
            )
            # continue to next frame instead of crashing
            continue

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.")
    except Exception as e:
        log_error("Unhandled error in application", e)
