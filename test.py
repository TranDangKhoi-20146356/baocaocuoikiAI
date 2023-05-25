

import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from keras.utils import load_img, img_to_array
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import load_img, img_to_array
import os
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
import os
import numpy as np 

from keras.models import load_model
from keras.utils import load_img, img_to_array
import os
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
import cv2
import os

import cv2
import os

def capture_and_save_images(folder_path, num_images):
    # Tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Khởi tạo camera
    camera = cv2.VideoCapture(0)

    count = 1

    while count <= num_images:
        # Đọc từng khung hình
        ret, frame = camera.read()

        # Hiển thị khung hình
        cv2.imshow("Capture Image", frame)

        # Chờ phím nhấn
        key = cv2.waitKey(1)

        # Khi nhấn phím 's', lưu ảnh vào thư mục
        if key == ord('s'):
            image_name = f"captured_image_{count}.jpg"
            image_path = os.path.join(folder_path, image_name)
            cv2.imwrite(image_path, frame)
            print(f"Captured image {count} saved as {image_name}")
            count += 1

        # Khi nhấn phím 'q', thoát khỏi chương trình
        if key == ord('q'):
            break

    # Giải phóng bộ nhớ và đóng camera
    camera.release()
    cv2.destroyAllWindows()

# Gọi hàm để chụp và lưu nhiều ảnh
folder_path = "C:/Users/MSI/Desktop/Face-Mask-Detection-master1/Face-Mask-Detection-master/test"
num_images = 5 # Số lượng ảnh muốn chụp và lưu
capture_and_save_images(folder_path, num_images)






train_data=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_set=train_data.flow_from_directory('test',target_size=(150,150), batch_size=64, class_mode='categorical')
test_data= ImageDataGenerator(rescale = 1./255)
test_set.class_indices

model_1=load_model('Hoa(CNN).h5')

test='test'

for i in os.listdir(test):
  img=load_img(test+'/'+i,target_size=(150,150))
  plt.imshow(img)
  img=img_to_array(img) 
  img=img.astype('float32') 
  img=img/255 
  img=np.expand_dims(img,axis=0)
  result=(model_1.predict(img).argmax()) 
  class_name=['mask','without_mask']
  print(class_name[result]) 
  plt.show()




  