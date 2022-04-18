import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet import model as retinanet
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerRetinaNet
from pytorch_grad_cam.utils.model_targets import RetinaNetBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import retinanet_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

import torch.nn as nn
import torchvision

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


def detect_image(parser):
    image_dir = parser.image_dir
    depth = parser.depth
    ckpt = parser.ckpt
    conf = parser.conf
    class_list = parser.class_list
    save = parser.save_images
    show = parser.show_images
    demo = parser.demo

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

    model.backbone.load_state_dict(torch.load(
        ckpt, map_location='cpu'), strict=False)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    target_layers = [model.backbone.fpn]

    def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(
                grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(
            image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    for img_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, img_name))
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

        new_image = cv2.resize(image, (640, 640))
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        image = torch.from_numpy(image).float()
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        outputs = model(image.float())[0]
        scores, classification, transformed_anchors = outputs[
            'scores'], outputs['labels'], outputs['boxes']
        boxes, lbs = [], []
        idxs = np.where(scores.cpu() > conf)

        for j in range(idxs[0].shape[0]):
            boxes.append(transformed_anchors[idxs[0][j], :].detach(
            ).cpu().numpy().astype(np.int32))
            lbs.append(classification[idxs[0][j]])

        if parser.demo == 'eigencam':
            cam = EigenCAM(model,
                        target_layers,
                        use_cuda=torch.cuda.is_available(),
                        reshape_transform=retinanet_reshape_transform)
        elif parser.demo == 'ablationcam':
            # pass
            cam = AblationCAM(model,
                        target_layers, 
                        use_cuda=torch.cuda.is_available(), 
                        reshape_transform=retinanet_reshape_transform,
                        ablation_layer=AblationLayerRetinaNet())

        targets = [RetinaNetBoxScoreTarget(
            labels=lbs, bounding_boxes=boxes)]
        grayscale_cam = cam(image, targets=targets)[0, :]
        cam_image = show_cam_on_image(
            new_image / 255., grayscale_cam, use_rgb=True)
        
        # focus in all dog
        cam_image = renormalize_cam_in_bounding_boxes(boxes, new_image / 255., grayscale_cam)

        print('Elapsed time: {}'.format(time.time() - st))
        for i, box in enumerate(boxes):
            label_name = labels[lbs[i].item()]
            caption = '{}'.format(label_name)
            x1, y1, x2, y2 = box
            draw_caption(cam_image, (x1, y1, x2, y2), caption)
            cv2.rectangle(cam_image, (x1, y1), (x2, y2),
                              color=(0, 0, 255), thickness=2)

        if save:
            if not os.path.exists('gradcam'):
                os.mkdir('gradcam')
            cv2.imwrite(os.path.join('gradcam', img_name), cam_image)
        if show:
            cv2.imshow('gradcam', cam_image)
            cv2.waitKey(0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple script for visualizing and saving result of training.')

    parser.add_argument(
        '--image_dir', default='images', help='Path to directory containing images')
    parser.add_argument(
        '--depth', '-d', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument(
        '--ckpt', '-c', help='Path to model state_dict file', required=True)
    parser.add_argument(
        '--class_list', default='class_list.csv', help='Path to CSV file listing class names')
    parser.add_argument('--conf', help='Confidence threshold',
                        type=float, default=0.5)
    parser.add_argument(
        '--save_images', help='Save images with detections', action='store_true', default=True)
    parser.add_argument('--show_images', help='Show images with detections',
                        action='store_true', default=False)
    parser.add_argument('--demo', choices=['ablationcam', 'eigencam'], default='eigencam', help='Demo mode')

    parser = parser.parse_args()

    detect_image(parser)

# python gradcam_show.py --image_dir images --depth 50 --ckpt coco_resnet_50_map_0_335_state_dict.pt --class_list class_list.csv
