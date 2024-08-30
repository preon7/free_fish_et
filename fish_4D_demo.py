import os
import argparse
from src.extract_frames import *
from src.multiview_reconstruction import reconstruct
from src.smooth_eye_detection import detect_eye

def reconstruct_place(place, n_frames, data_folder, out_folder):
    reconstruct_args.fish_place = place
    reconstruct_args.datadir = data_folder
    reconstruct_args.outdir = out_folder

    if not os.path.exists(reconstruct_args.outdir):
        os.makedirs(reconstruct_args.outdir, exist_ok=True)

    reconstruct_args.index = list(range(n_frames))
    reconstruct(reconstruct_args)

if __name__ == '__main__':
    videos = ['data/videos/front.mp4',
              'data/videos/bottom.mp4']

    model_path = 'models/trained_models/kn_segmentation_model_2022-01-12_50'
    out_path = 'data/input_frames'
    video_folder = 'video_frames'

    input_folder = os.path.join(out_path, video_folder)
    out_folder = "data/output/"

    # ====== data processing ======
    print('extracting frames from video...')
    extract_from_video(videos, out_path, video_folder)

    print('processing frames from video...')
    process_input_folder(input_folder)

    print('detecting fish masks...')
    predict(input_folder, model_path, 'cuda')

    print('detecting fish keypoints...')
    detect_dlc(input_folder)

    # ==============================

    print('reconstructing mesh model sequence...')
    parser2 = argparse.ArgumentParser()
    reconstruct_args = parser2.parse_args()
    reconstruct_args.mesh = 'goldfish_design_small.json'
    reconstruct_args.fish_place = 2
    reconstruct_args.seed = 700
    reconstruct_args.save_models = False

    n_frames = 0

    with open(os.path.join(input_folder, 'index.json')) as jf:
        video_meta = json.load(jf)

    n_frames = video_meta['image_count']

    reconstruct_place(2, n_frames, input_folder, out_folder)
    reconstruct_place(1, n_frames, input_folder, out_folder)

    # ==============================

    print('detecting eye position...')

    parser = argparse.ArgumentParser()
    eye_args = parser.parse_args()

    eye_args.dir = os.path.join(out_folder, 'pose_pickle')
    eye_args.index_range = f'0-{n_frames}'
    eye_args.datadir = input_folder
    eye_args.eye_model = 'models/trained_models/eye_detection_model_2022-06-21'
    eye_args.fish_place = 2
    eye_args.epoches = 150
    eye_args.learning_rate = 1e-3

    detect_eye(eye_args)

    print('eye detection done')