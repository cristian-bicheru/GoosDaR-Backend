from PIL import Image
import random
import os

BASE_OCCLUSION_FACTOR = 0.25
OCCLUSION_RANDOM_RANGE = 0.25
NUM_CYCLES = 20
GAUSSIAN_NOISE_ALPHA = 0.1
GAUSSIAN_RANDOM_RANGE = 0.1
GAUSSIAN_NOISE_SIGMA = 1000
OCCLUSION_FILES = [Image.open('filters/'+x).convert("RGB") for x in os.listdir('filters')]

def occlude(im, ocm):
    noise = Image.effect_noise(im.size, GAUSSIAN_NOISE_SIGMA)
    blended = Image.blend(im, ocm.resize(im.size), BASE_OCCLUSION_FACTOR + (random.random()-1)*(OCCLUSION_RANDOM_RANGE/0.5))
    return Image.blend(blended, noise.convert("RGB"), GAUSSIAN_NOISE_ALPHA + (random.random()-1)*(GAUSSIAN_RANDOM_RANGE/0.5))

def run(d):
    pidx = 0
    for _ in range(NUM_CYCLES):
        for file in os.listdir('images/'+d):
            occlude(Image.open('images/'+d+'/'+file).convert("RGB"), random.choice(OCCLUSION_FILES)).save('oimages/'+d+'/'+str(pidx)+'.jpg')
            with open("annotations/"+d+'/'+file.split('.')[0]+'.xml', 'r') as f:
                data = f.readlines()
            for i in range(len(data)):
                if "<path>" in data[i]:
                    data[i] = data[i].replace("images", "oimages").replace(file, str(pidx)+'.jpg')
                    break
            with open("oannotations/"+d+'/'+str(pidx)+'.xml', 'w') as f:
                f.write(''.join(data))
            pidx += 1

run("train")
run("test")
