import os
import argparse

from src.phenotype import phenotype
from src.genotype import Genotype
from src.evolver import evolve
from src.display import display, load_img

import numpy as np

parser = argparse.ArgumentParser(prog='ShapePainter', description='Watch an evolutionary network learn draw your image using just 256 different shapes and coloured circles.')

parser.add_argument('targetfile')
parser.add_argument("-C", "--checkpointfile", default=None)

class Application:
    def __init__(self, args):
        self.targetfile = args.targetfile
        self.checkpointfile = args.checkpointfile


    def load_fake_image(self):
        s = np.zeros((256,7))

        s[0,:] = [1  ,0.5,0.5,0.5,0.5,1,   0]
        s[1,:] = [0.5,0.5,0.6,0.5,0.5,1,   0]
        s[2,:] = [1  ,0  ,0.7,0.5,0.5,1,   0]
        s[3,:] = [1  ,0.5,0.5,0.6,0.5,0.5, 0]
        s[4,:] = [1  ,0.5,0.6,0.6,0.5,0.25,0]
        s[5,:] = [1  ,0.5,0.7,0.6,1,  0.25,0]
        s[6,:] = [1  ,0.5,0.8,0.6,1,1,     0]

        img = phenotype(Genotype(genes=s), image_data=(500,500))

        display(img)


    def start(self):
        # Initialize target image
        target = load_img(self.targetfile, size=(100,100))

        checkpoint = np.load(self.checkpointfile) if self.checkpointfile else None

        evolve(target, checkpoint)

        return 0

if __name__ == "__main__":
    args = parser.parse_args()

    application = Application(args)
    application.start()
