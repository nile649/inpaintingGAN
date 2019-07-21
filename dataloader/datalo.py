import torchvision.transforms.functional as TF
import torchvision
from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import sobel
from skimage.exposure import rescale_intensity
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
import numpy as np

class testdata():
    """Dataset for Inpainting.
    Task: To load saved edge map and save destroyed image.
    """
    def __init__(self):
    
        self.size = 128
    
    def transformData(self, cimg, eimg):

        src = TF.to_tensor(cimg)
        Esrc= TF.to_tensor(eimg)

        src = TF.normalize(src,(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        Esrc = TF.normalize(Esrc,(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return src,Esrc

    
    def getImages(self, cimg_path,eimg_path):
        cimg = rescale_intensity(plt.imread(cimg_path)/255)
        eimg = rescale_intensity(plt.imread(eimg_path)/255)

        cimg = resize(cimg,(128,128))
        eimg = resize(eimg,(128,128))
        eimg = eimg[:,:,0]
        eimg = np.expand_dims(eimg,-1)
        
        
        cimg = Image.fromarray(np.uint8(cimg*255))
   
        source,Esource = self.transformData(cimg,eimg)

        return source.unsqueeze(0),Esource.unsqueeze(0)
