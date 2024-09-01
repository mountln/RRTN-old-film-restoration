# RRTN-old-film-restoration

This repository contains the code for the paper [Restoring Degraded Old Films with Recursive Recurrent Transformer Networks](https://ieeexplore.ieee.org/document/10483892).

Much of this code is based on the prior study [Bringing Old Films Back to Life](http://raywzy.com/Old_Film/).
Additionally, parts of the model code are based on [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++/).
For more details, please refer to the papers of these studies.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/mountln/RRTN-old-film-restoration.git
    ```

1. Install the required dependencies:

    ```bash
    conda env create -f environment.yml
    conda activate rrtn
    mim install mmcv  # install mmcv
    ```

### Restore Old Films

1. Download the pretrained models ([
rrtn_128_first.pth](https://github.com/mountln/RRTN-old-film-restoration/releases/download/latest/rrtn_128_first.pth), [
rrtn_128_second.pth](https://github.com/mountln/RRTN-old-film-restoration/releases/download/latest/rrtn_128_second.pth), [
raft-sintel.pth](https://github.com/mountln/RRTN-old-film-restoration/releases/download/latest/raft-sintel.pth)). Save them in the `./pretrained_models` directory.

1. (Optional) Remove duplicate frames. This step is optional, but some old films downloaded from the Internet may contain duplicate frames. Removing these duplicate frames can sometimes improve the results. You can use `ffmpeg` to do this by referring to [here](https://stackoverflow.com/questions/37088517/remove-sequentially-duplicate-frames-when-using-ffmpeg). There are also other methods available for removing duplicate frames.

1. Prepare the video to be processed by following the directory structure shown in `test_data_sample/`. The folder structure should be as follows:

    ```
    test_data_sample
    ├── video_1
    │   ├── 00001.png
    │   ├── 00002.png
    │   ├── ...
    │   └── xxxxx.png
    ├── ...
    └── video_n
        ├── 00001.png
        ├── 00002.png
        ├── ...
        └── xxxxx.png
    ```

1. Run `restore.py` to restore the videos. For example, to restore videos in the `test_data_sample/` folder, use the following command:

    ```bash
    python VP_code/restore.py --name rrtn --model_name rrtn --model_path_first pretrained_models/rrtn_128_first.pth --model_path_second pretrained_models/rrtn_128_second.pth --temporal_length 30 --temporal_stride 15 --input_video_url test_data_sample
    ```

    If GPU memory is not enough, try reducing `temporal_length`, `temporal_stride`, or the resolution of the input video.

### Train Models

1. Download the [noise data](https://github.com/mountln/RRTN-old-film-restoration/releases/download/latest/noise_data.zip).
Alternatively, you can download the same noise data from the links provided by [Bringing Old Films Back to Life](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life?tab=readme-ov-file#usage) or [DeepRemaster](https://github.com/satoshiiizuka/siggraphasia2019_remastering?tab=readme-ov-file#dataset)

1. Download the [REDS dataset](https://seungjunnah.github.io/Datasets/reds.html).

1. Modify the paths in `configs/rrtn.yaml`.

1. To train the model for the first recursion, use the following command:

    ```bash
    python VP_code/main_gan.py --name rrtn --num_recursion 1 --epoch 30 --gpus 4
    ```

1. To train the model for further recursion, use the following command:

    ```bash
    python VP_code/main_gan.py --name rrtn --num_recursion 2 --epoch 30 --gpus 4
    ```
