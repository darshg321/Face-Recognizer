# Face-Recognizer

Detect specific faces from a videostream and save unknown faces using facial recognition.

To use:

1. `git clone https://github.com/darshg321/poseidon.git`
2. `cd poseidon`
3. `pip install -r requirements.txt`
4. `python poseidon.py`

Green box around a face means it's in sample_images, red means it's new
Known faces will be saved with name of the original face in sample_images + number, unknown faces will be saved with unknown + number

Args:
--images_path - The path to the sample images folder
--video_path - The video file. Supported formats: .mkv, .mp4, .avi, .mov, .wmv
--min_probability - The minimum probability for a face to be detected. Default is 0.95, max is 1. Higher values mean more strict detection.
--max_distance - The threshold for the distance of faces when comparing faces. Higher values mean looser comparing, with a max of 1. Default is 0.4