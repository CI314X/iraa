import cv2    
from sys import argv
import numpy as np
from tqdm import tqdm
import skvideo.io

if __name__ == '__main__':
    input_file = argv[1]
    output_file = argv[2]
    noise_file = argv[3]
    eps = float(argv[4])

    video_data = skvideo.io.vread(input_file, 1080,1920, inputdict={'-pix_fmt':'yuv420p'})
    video_length = video_data.shape[0]
    video_channel = video_data.shape[3]
    video_height = video_data.shape[1]
    video_width = video_data.shape[2]

    universal_noise = cv2.imread(noise_file)
    universal_noise = cv2.cvtColor(universal_noise, cv2.COLOR_BGR2RGB).astype('float32')
    universal_noise /= 255.
    universal_noise -= 0.5
    universal_add = np.tile(universal_noise,(1080//256 + 1, 1920//256 + 1, 1))[:1080, :1920, :]
    universal_add[universal_add>eps] = eps
    universal_add[universal_add<-eps] = -eps
 
    for s in tqdm(range(video_length)):
        frame = video_data[s].astype('float32') / 255.
        res = (frame + universal_add)
        res[res < 0] = 0
        res[res > 1] = 1.
        res = (res * 255).astype('uint8')
        video_data[s]=res

    skvideo.io.vwrite(output_file,video_data,outputdict={'-pix_fmt':'yuv420p'})
    
    