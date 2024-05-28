from common_imports import os
import argparse

def cl_args():
    parser = argparse.ArgumentParser(description='Find faces in a video and compare them to images of faces.\nSupported image formats: .jpg, .jpeg, .png.')
    parser.add_argument('--images_path', type=str, help='The path to the sample images folder')
    parser.add_argument('--load_amount', type=int, help='The amount of images to load after filtering out unsupported formats. Default is 10.')
    parser.add_argument('--video_path', type=str, help='The video file. Supported formats: .mkv, .mp4, .avi, .mov, .wmv.')
    parser.add_argument('--min_probability', type=float, help='The minimum probability for a face to be detected. Default is 0.95.\
    \nMax is 1. Higher values mean more strict detection.')
    parser.add_argument('--max_distance', type=float, help='The threshold for the distance of faces when comparing faces.\
    \nHigher values mean looser comparing, with a max of 1. Default is 0.4.')

    args = parser.parse_args()
    
    images_to_load = args.load_amount
    images_path = args.images_path
    video_file = args.video_path
    min_probability = args.min_probability
    max_distance = args.max_distance
    
    allowed_video_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.wmv']
    
    # if all args are not specified, use example values
    if all(not arg for arg in [args.images_path, args.load_amount, args.video_path, args.min_probability, args.max_distance]):
        # maybe change these
        images_to_load = 10
        images_path = './sample_images/'
        video_file = './faceexamplevideo.mkv'
        min_probability = 0.95
        max_distance = 0.4
        print('Args not specified, using example values')
    else:
        # if an arg is specified, but not all, print error and exit
        if not all(arg for arg in [args.images_path, args.load_amount, args.video_path]):
            parser.error('Missing required arguments')
    
    if not args.min_probability:
        min_probability = 0.95
        
    if not args.max_distance:
        max_distance = 0.4
    
    if not os.path.exists(video_file):
        raise Exception('Video file does not exist')
    if not any(video_file.endswith(ext) for ext in allowed_video_extensions):
        raise Exception('Video file extension not supported')
    
    return images_to_load, images_path, video_file, min_probability, max_distance