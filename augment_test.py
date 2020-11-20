import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    sometimes(iaa.Affine(scale={"x": (0.3, 1), "y": (0.3, 1)}, shear=(-60, 60), rotate=(-20, 20), cval=255)),
    sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1), keep_size=False, cval=255)),
    iaa.AddToHueAndSaturation(value=(-50, 50)),
    iaa.Fliplr(0.5),
    iaa.Affine(scale=(0.2, 0.5), cval=255)], random_order=True)


def draw_LP_by_vertices(img, pts, color=(255, 0, 234) , cls = "p"):
    cv2.polylines(img, [np.array(pts)], isClosed=True, color=color, thickness=1)
    p = [np.array(pts)]
    x , y = p[0][0][0] , p[0][0][1]
    img = cv2.putText(img, str(cls), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return img


def BBCor_to_pts(pt1, pt2):
    x_min = min(pt1[0], pt2[0])
    x_max = max(pt1[0], pt2[0])
    y_min = min(pt1[1], pt2[1])
    y_max = max(pt1[1], pt2[1])
    return [[x_max, y_max], [x_min, y_max], [x_min, y_min], [x_max, y_min]]


def display(image, points):
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    color = (255, 0, 0)
    thickness = 2
    return cv2.polylines(image, [pts], isClosed, color, thickness)


images_path = "D:/labs/ssd_model_train/anpr_det_12/"
ann_path = "D:/labs/ssd_model_train/anpr_det_12/"
out_dir = "./aug_single_data_several_times"
use_four_pts = False

def aug_on_images(file_name):
    ann_dir_txt_path = os.path.join(ann_path, file_name)
    image_dir_path = os.path.join(images_path, '.'.join(file_name.split('.')[:-1]) + '.jpg')
    file_txt = open(ann_dir_txt_path, 'r')
    image = cv2.imread(image_dir_path)
    h, w, _ = image.shape
    Ann_list = []
    Image_list = []
    points = []
    class_list = []
    for f in file_txt:
        f = f.replace("\n", "").split(',')
        if use_four_pts:
            X = []
            Y = []
            class_list.append(f[-2])
            for p in f[1:-6]:
                X.append(float(p) * w)
            for p in f[-6:-2]:
                Y.append(float(p) * h)
            for x, y in zip(X, Y):
                points.append([int(x), int(y)])  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        else:
            x1 , y1 , x2 , y2 = int(f[0]) , int(f[1]) , int(f[2]), int(f[3])
            class_list.append(f[-1])
            f_points = BBCor_to_pts((x1 , y1) , (x2 , y2))
            for pts in f_points:
                points.append(pts)


    Ann_list.append(np.array(points))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image_list.append(image)
    imgs_aug, vertices_aug = seq(images=Image_list, keypoints=Ann_list)
    for img_aug in imgs_aug:
        cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
    return imgs_aug, vertices_aug, class_list


n = 1

for file_name in os.listdir(ann_path):
    if file_name.endswith('.txt'):
        idx = 0
        imgs_aug, vertices_aug, class_list = aug_on_images(file_name)
        print(class_list)
        for image_aug, keypoint_aug in zip(imgs_aug, vertices_aug):
            image_aug = draw_LP_by_vertices(image_aug, keypoint_aug[0:4] , cls = class_list[idx])
            for i in range(4, len(keypoint_aug), 4):
                idx = idx + 1
                image_aug = draw_LP_by_vertices(image_aug, keypoint_aug[i:i + 4], (36, 247, 255) , cls = class_list[idx])
            n += 1
            cv2.imwrite(os.path.join(out_dir, '%d.jpg' % n), image_aug)
        #break
