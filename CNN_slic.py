from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import datetime
from PIL import Image
import numpy as np

imgname="taili"
image = img_as_float(io.imread(imgname+".png"))
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+" Start.")

for numSegments in [8000]:#1000,2000,3000,4000,5000,6000,7000,9000,10000
    for cp in [5]:#3,4,6,2
        for sig in [6]:#2,4,6,
            segments = slic(image, n_segments = numSegments, sigma = sig,compactness=cp)

            img=Image.fromarray(np.array(segments, np.uint8))
            img.save(imgname+"_%d seg_" % (numSegments)+str(cp)+"_comp"+"_%d_sigma.png" % (sig) , "png")

            print(imgname+"_%d bodr" % (numSegments)+str(cp)+"_comp"+"_%d_sigma.png " % (sig)+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ')+" Output over.")

