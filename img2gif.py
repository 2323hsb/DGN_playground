import imageio
images = []
import os
roc = './results/vae/ring'
filenames = os.listdir(roc)
for filename in filenames:
    images.append(imageio.imread(roc+"/"+filename))
imageio.mimsave('./results/gifs/vae.gif', images, duration=0.03)