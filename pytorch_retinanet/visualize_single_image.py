import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet import model as retinanet

print('CUDA available: {}'.format(torch.cuda.is_available()))


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError(
                'line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError(
                'line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, depth, ckpt, conf, class_list, save, show):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    if depth == 18:
        model = retinanet.resnet18(pretrained=False, num_classes=len(classes))
    elif depth == 34:
        model = retinanet.resnet34(pretrained=False, num_classes=len(classes))
    elif depth == 50:
        model = retinanet.resnet50(pretrained=False, num_classes=len(classes))
    elif depth == 101:
        model = retinanet.resnet101(pretrained=False, num_classes=len(classes))
    elif depth == 152:
        model = retinanet.resnet152(pretrained=False, num_classes=len(classes))
    else:
        raise ValueError('depth must be 18, 34, 50, 101, or 152')

    model.backbone.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        # image = cv2.resize(image, (int(round(cols * scale)),
        #                    int(round((rows * scale)))))
        # rows, cols, cns = image.shape

        # pad_w = 32 - rows % 32
        # pad_h = 32 - cols % 32

        # new_image = np.zeros(
        #     (rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        # new_image[:rows, :cols, :] = image.astype(np.float32)
        new_image = cv2.resize(image, (640, 640))
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            # print(image.shape, image_orig.shape, scale)
            outputs = model(image.float())[0]
            scores, classification, transformed_anchors = outputs['scores'], outputs['labels'], outputs['boxes']
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > conf)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / new_image.shape[1] * image_orig.shape[1])
                y1 = int(bbox[1] / new_image.shape[0] * image_orig.shape[0])
                x2 = int(bbox[2] / new_image.shape[1] * image_orig.shape[1])
                y2 = int(bbox[3] / new_image.shape[0] * image_orig.shape[0])
                label_name = labels[int(classification[idxs[0][j]])]
                score = scores[idxs[0][j]]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2),
                              color=(0, 0, 255), thickness=2)

            if save:
                if not os.path.exists('detections'):
                    os.mkdir('detections')
                cv2.imwrite(os.path.join('detections', img_name), image_orig)
            if show:
                cv2.imshow('detections', image_orig)
                cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple script for visualizing and saving result of training.')

    parser.add_argument(
        '--image_dir', help='Path to directory containing images')
    parser.add_argument(
        '--depth', '-d', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument(
        '--ckpt', '-c', help='Path to model state_dict file', required=True)
    parser.add_argument(
        '--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument('--conf', help='Confidence threshold',
                        type=float, default=0.5)
    parser.add_argument(
        '--save_images', help='Save images with detections', action='store_true', default=True)
    parser.add_argument('--show_images', help='Show images with detections',
                        action='store_true', default=False)

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.depth, parser.ckpt, parser.conf,
                 parser.class_list, parser.save_images, parser.show_images)
