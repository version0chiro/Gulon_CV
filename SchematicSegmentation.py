import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
ctx = mx.gpu(0)

url = 'https://github.com/zhanghang1989/image-data/blob/master/encoding/' + \
    'segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
filename = 'ade20k_example.jpg'
gluoncv.utils.download(url, filename, True)

img = image.imread(filename)

from matplotlib import pyplot as plt
plt.imshow(img.asnumpy())
plt.show()

from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img,ctx)

model = gluoncv.model_zoo.get_model('deeplab_resnest101_ade', pretrained=True)

import time
start = time.time()
output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
print(time.time() - start)

from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'ade20k')
mask.save('output.png')

mmask = mpimg.imread('output.png')
plt.imshow(mmask)
plt.show()