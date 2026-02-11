from __future__ import annotations

import argparse
import os
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

def cl_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find faces in a video and compare them to images of faces.\n"
            "Supported image formats: .jpg, .jpeg, .png."
        )
    )
    parser.add_argument(
        "--images_path",
        type=str,
        help="The path to the sample images folder",
    )
    parser.add_argument(
        "--load_amount",
        type=int,
        help=(
            "The amount of images to load after filtering out unsupported "
            "formats. Default is 10."
        ),
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="The video file. Supported formats: .mkv, .mp4, .avi, .mov, .wmv.",
    )
    parser.add_argument(
        "--use_webcam",
        action="store_true",
        help="Use the default webcam instead of a video file. If set, --video_path is ignored.",
    )
    parser.add_argument(
        "--min_probability",
        type=float,
        help=(
            "The minimum probability for a face to be detected. Default is 0.95.\n"
            "Max is 1. Higher values mean more strict detection."
        ),
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        help=(
            "The threshold for the distance of faces when comparing faces.\n"
            "Higher values mean looser comparing, with a max of 1. Default is 0.4."
        ),
    )
    parser.add_argument(
        "--min_live_area",
        type=int,
        help=(
            "Minimum face area (in pixels^2) required to add a new, "
            "previously unrecognized face into live_detected. "
            "Default is 4900 (70x70)."
        ),
    )

    args = parser.parse_args()

    use_webcam = args.use_webcam
    images_to_load = args.load_amount
    images_path = args.images_path
    video_file = args.video_path
    min_probability = args.min_probability
    max_distance = args.max_distance
    min_live_area = args.min_live_area

    allowed_video_extensions = [".mkv", ".mp4", ".avi", ".mov", ".wmv"]

    # if no config args are specified, use example values
    user_provided_any_config = any(
        arg is not None
        for arg in [
            args.images_path,
            args.load_amount,
            args.video_path,
            args.min_probability,
            args.max_distance,
            args.min_live_area,
        ]
    )

    if not user_provided_any_config:
        images_to_load = 10
        images_path = "./sample_images/"
        # default source: example video, unless webcam explicitly requested
        video_file = 0 if use_webcam else "./faceexamplevideo.mkv"
        min_probability = 0.95
        max_distance = 0.4
        min_live_area = 70 * 70
        print("Args not specified, using example values")
    else:
        # some config was provided -> enforce required ones
        if images_path is None or images_to_load is None or (
            video_file is None and not use_webcam
        ):
            parser.error("Missing required arguments")
        if use_webcam:
            video_file = 0

    if min_probability is None:
        min_probability = 0.95

    if max_distance is None:
        max_distance = 0.4

    if min_live_area is None:
        min_live_area = 70 * 70

    # validate only when using a file, not webcam
    if not use_webcam:
        if not os.path.exists(video_file):
            raise Exception("Video file does not exist")
        if not any(video_file.endswith(ext) for ext in allowed_video_extensions):
            raise Exception("Video file extension not supported")

    return (
        images_to_load,
        images_path,
        video_file,
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
        and (
            any(file.endswith(ext) for ext in allowed_image_extensions)
            or print(f"File format not supported: {file}")
        )
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
            print(
                f"Loaded face {len(images_embeddings)}: file {images_path + file}"
            )
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
        ) = cl_args()
    except Exception as e:
        log_error("Failed to parse command-line arguments", e)
        return

    try:
        embeddings = load_embeddings(images_to_load, images_path)
    except Exception as e:
        log_error("Failed to load embeddings", e)
        embeddings = {}

    print("Finished loading faces")

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
                print(
                    f"Error loading video or video file ended (Frame: {frame_count})"
                )
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
                        print(
                            f"A face matched with {cosine_similarity}% distance "
                            f"from embedding {embedding_index} in list (File: {match_file})"
                        )
                        print(f"Match found at frame {frame_count}")
                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                        )
                    else:
                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), (0, 0, 255), 2
                        )
                        # Unrecognized face: only add to live_detected if it passes
                        # quality checks (size, sharpness, rough frontal angle).
                        # We still always *try* to recognize every face above via
                        # embeddings, even if we don't store it.
                        if is_face_high_quality_for_live_detect(
                            face, face_area, min_live_area
                        ):
                            face_file = save_unrecognized_face_and_add_embedding(
                                face, face_embedding, embeddings
                            )
                            if face_file:
                                print(
                                    f"Unrecognized face saved to {face_file} "
                                    f"(area={face_area}px²)"
                                )
                        else:
                            print(
                                "Unrecognized face rejected for live_detect "
                                f"(area={face_area}px², min_area={min_live_area}px², "
                                "likely blurry/partial/profile)."
                            )
                    if match_info:
                        # Optionally keep saving recognized faces as before
                        face_file = save_face(face, match_file)
                        if face_file:
                            print(f"Face saved to {face_file}")

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
