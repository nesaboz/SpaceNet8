from PIL import Image
import time
tic = time.time()
# filepath = '/home/paperspace/Downloads/10300100AF395C00_2_13_45.tif'
filepath = '/tmp/share/data/spacenet8/Louisiana-East_Training_Public/PRE-event/10300100AF395C00_2_13_45.tif'
for i in range(100):
    Image.open(filepath)
toc = time.time()
print(toc - tic)
