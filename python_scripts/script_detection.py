import sys
import argparse
from pathlib import Path

from collections import namedtuple

import numpy as np
import tensorflow as tf
import cv2

from anchors import Anchor, load_face_anchors

import torch
sys.path.append('../')
sys.path.append('../open_closed_eye')
from open_closed_eye.train import Net

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

Rect = namedtuple('Rect', ['x', 'y', 'width', 'height'])
Point = namedtuple('Point', ['x', 'y'])
Detection = namedtuple('Detection', ['class_id', 'score', 'bounding_box', 'keypoints'])

RED_COLOR = (0, 0, 255, 255)
GREEN_COLOR = (0, 255, 0, 255)
BLUE_COLOR = (255, 0, 0, 255)
YELLOW_COLOR = (0, 255, 255, 255)


def resize_aspect_fit(src: np.array, target_side_size: int) -> tuple:
    src_height, src_width = src.shape[:2]
    max_side = max(src_width, src_height)
    scale = target_side_size / max_side
    target_width = int(round(scale * src_width))
    target_height = int(round(scale * src_height))
    interpolation_type = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    dst = cv2.resize(src, (target_width, target_height), interpolation=interpolation_type)
    top_border = int((target_side_size - target_height) / 2)
    bottom_border = target_side_size - target_height - top_border
    left_border = int((target_side_size - target_width) / 2)
    right_border = target_side_size - target_width - left_border
    dst = cv2.copyMakeBorder(dst, top_border, bottom_border, left_border,
                             right_border, cv2.BORDER_CONSTANT)
    return dst, Rect(left_border, top_border, target_width, target_height)


class BlazeFace:
    def __init__(self, model_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        # Parameters are taken from the following graph
        # https://github.com/google/mediapipe/blob/master/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.number_of_keypoints = 6
        self.number_of_boxes = 896
        self.number_of_coordinates = 16
        self.number_of_classes = 1
        self.box_coordinates_offset = 0
        self.keypoints_coordinates_offset = 4
        self.number_of_values_per_keypoint = 2
        self.min_score_threshold = 0.5
        self.min_suppression_threshold = 0.4
        self.score_clipping_threshold = 100.
        self.use_score_clipping_threshold = True
        self.use_sigmoid_score = True
        self.is_input_vertically_flipped = False
        self.is_output_is_reversed = True
        input_details = self.interpreter.get_input_details()
        self.input_tensor_index = input_details[0]['index']
        self.input_size = input_details[0]['shape'][1]
        output_details = self.interpreter.get_output_details()
        self.regressors_output_index = output_details[0]['index']
        self.classificators_output_index = output_details[1]['index']
        self.x_scale = self.input_size
        self.y_scale = self.x_scale
        self.h_scale = self.x_scale
        self.w_scale = self.x_scale
        self.anchors = load_face_anchors()

    def detect(self, image: np.array) -> list:
        # Preprocess input image
        working_image, roi = self._preprocess(image)
        # Run inference
        self._inference(working_image)
        # Decode results
        return self._postprocess(image.shape[1], image.shape[0], roi)

    def _preprocess(self, image: np.array) -> tuple:
        working_image, roi = resize_aspect_fit(image, self.input_size)
        working_image = (working_image.astype(np.float32) - 127.5) / 127.5
        return np.resize(working_image, (1, *working_image.shape)), roi

    def _inference(self, image: np.array):
        self.interpreter.set_tensor(self.input_tensor_index, image)
        self.interpreter.invoke()

    def _postprocess(self, image_width: int, image_height: int, roi: Rect, ) -> list:
        boxes, keypoints = self._decode_boxes_and_keypoints()
        scores, classes = self._decode_score_and_classes()
        resulting_boxes_indices = cv2.dnn.NMSBoxes(boxes, scores, self.min_score_threshold, self.min_suppression_threshold)
        indices = np.squeeze(resulting_boxes_indices, axis=-1) if resulting_boxes_indices else []
        detections = []
        for index in indices:
            box = boxes[index]
            bounding_box = Rect(
                int(max(image_width * (self.input_size * box.x - roi.x) / roi.width, 0)),
                int(max(image_height * (self.input_size * box.y - roi.y) / roi.height, 0)),
                int(image_width * (self.input_size * box.width) / roi.width),
                int(image_height * (self.input_size * box.height) / roi.height)
            )
            detection_keypoints = [
                Point(
                    int(image_width * (self.input_size * keypoint.x - roi.x) / roi.width),
                    int(image_height * (self.input_size * keypoint.y - roi.y) / roi.height)
                )
                for keypoint in keypoints[index]
            ]
            detections.append(Detection(
                classes[index], scores[index], bounding_box, detection_keypoints
            ))
        return detections

    def _decode_boxes_and_keypoints(self) -> tuple:
        regressors_output = self.interpreter.get_tensor(self.regressors_output_index)
        if self.is_output_is_reversed:
            box_offset_y = 1
            box_offset_x = 0
            box_offset_height = 3
            box_offset_width = 2
            keypoint_offset_y = 1
            keypoint_offset_x = 0
        else:
            box_offset_y = 0
            box_offset_x = 1
            box_offset_height = 2
            box_offset_width = 3
            keypoint_offset_y = 0
            keypoint_offset_x = 1
        boxes = []
        keypoints = []
        for regression_data, anchor in zip(regressors_output[0], self.anchors):
            center_y = regression_data[self.box_coordinates_offset +
                                       box_offset_y] / self.y_scale * anchor.height + anchor.center_y
            center_x = regression_data[self.box_coordinates_offset +
                                       box_offset_x] / self.x_scale * anchor.width + anchor.center_x
            height = regression_data[self.box_coordinates_offset +
                                     box_offset_height] / self.h_scale * anchor.height
            width = regression_data[self.box_coordinates_offset +
                                    box_offset_width] / self.w_scale * anchor.width

            min_x = center_x - 0.5 * width
            min_y = center_y - 0.5 * height
            boxes.append(Rect(min_x, min_y, width, height))

            box_keypoints = []
            for i in range(self.number_of_keypoints):
                keypoint_idx = i * self.number_of_values_per_keypoint + self.keypoints_coordinates_offset
                keypoint_y = regression_data[keypoint_idx + keypoint_offset_y] / self.y_scale * anchor.height + anchor.center_y
                keypoint_x = regression_data[keypoint_idx + keypoint_offset_x] / self.x_scale * anchor.width + anchor.center_x
                box_keypoints.append(Point(keypoint_x, keypoint_y))
            keypoints.append(box_keypoints)
        return boxes, keypoints

    def _decode_score_and_classes(self) -> tuple:
        classificators_data = self.interpreter.get_tensor(self.classificators_output_index)
        class_scores = classificators_data[0]
        if self.use_score_clipping_threshold:
            class_scores = np.clip(class_scores, -self.score_clipping_threshold,
                                   self.score_clipping_threshold)
        if self.use_sigmoid_score:
            class_scores = 1.0 / (1.0 + np.exp(-class_scores))
        classes = np.argmax(class_scores, axis=1)
        scores = np.take_along_axis(class_scores,
                                    np.expand_dims(classes, -1), axis=1).squeeze(axis=-1)
        return scores, classes

def get_eyes(detection):

    eye_width = int(detection.bounding_box.width / 4)
    eye_height = int(detection.bounding_box.height / 4)

    left_eye_x = int(detection.keypoints[0].x - eye_width / 2)
    left_eye_y = int(detection.keypoints[0].y - eye_height / 2)

    right_eye_x = int(detection.keypoints[1].x - eye_width / 2)
    right_eye_y = int(detection.keypoints[1].y - eye_height / 2)

    left_eye_box = Rect(left_eye_x, left_eye_y, eye_width, eye_height)
    right_eye_box = Rect(right_eye_x, right_eye_y, eye_width, eye_height)

    return left_eye_box, right_eye_box

def crop_rect(frame, rect):
    return frame[rect.y : (rect.y + rect.height), rect.x : (rect.x + rect.width)]

def crop_eyes(frame, detection):
    left_eye, right_eye = get_eyes(detection)

    left_eye_frame = crop_rect(frame, left_eye)
    right_eye_frame = crop_rect(frame, right_eye)

    return left_eye_frame, right_eye_frame

def draw_detections(frame: np.array, detections: list, preds) -> np.array:
    for detection in detections:
        print(f'BOUNDING BOX: {detection.bounding_box}')
        print(f'WIDTH: {detection.bounding_box.width}')
        print(f'HEIGHT: {detection.bounding_box.height}')

        print(f'LEFT EYE: {detection.keypoints[0].x}')

        left_eye_box, right_eye_box = get_eyes(detection)

        frame = cv2.rectangle(frame, detection.bounding_box, YELLOW_COLOR, 2)
        if preds[0] > 0.5:
            frame = cv2.rectangle(frame, left_eye_box, GREEN_COLOR, 2)
        else:
            frame = cv2.rectangle(frame, left_eye_box, RED_COLOR, 2)
        if preds[1] > 0.5:
            frame = cv2.rectangle(frame, right_eye_box, GREEN_COLOR, 2)
        else:
            frame = cv2.rectangle(frame, right_eye_box, RED_COLOR, 2)

    return frame

def save_detections(output_dir: Path, frame_index: int, detections: list):
    fs = cv2.FileStorage(str(output_dir / f'frame_{frame_index:06}.json'), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print('Can not save detections for', frame_index)
        return
    fs.startWriteStruct('detections', cv2.FILE_NODE_SEQ)
    for detection in detections:
        fs.startWriteStruct('', cv2.FILE_NODE_MAP)
        fs.write('class', detection.class_id)
        fs.write('score', detection.score)
        fs.write('bounding_box', detection.bounding_box)
        fs.startWriteStruct('keypoints', cv2.FILE_NODE_SEQ)
        keypoint_count = 0
        for kp in detection.keypoints:

            fs.startWriteStruct('', cv2.FILE_NODE_MAP)
            fs.write('x', kp.x)
            fs.write('y', kp.y)
            fs.endWriteStruct()
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.release()


def parse_arguments():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--face_detection_model_path', type=lambda p: Path(p).absolute(),
                        default=Path(__file__).absolute().parent / 'models/face_detection_front.tflite',
                        help='Path to the models directory')
    parser.add_argument('--input', type=str, default=0,
                        help='Path to video source (device or video file)')
    parser.add_argument('--output', type=lambda p: Path(p).absolute(),
                        default=Path(__file__).absolute().parent / 'output',
                        help='Path to store output results')
    parser.add_argument('--record_video', action='store_true', help='Record result video file')
    parser.add_argument('--save_detections', action='store_true',
                        help='Store per frame results in output folder')
    parser.add_argument('--open_closed_weights', help='Path to open-closed model weights')
    return parser.parse_args()


def main():
    args = parse_arguments()
    should_create_output_folder = args.record_video or args.save_detections
    if should_create_output_folder:
        args.output.mkdir(parents=True)
        print(f'Results will be stored in the "{args.output}"')
    else:
        print('Run does not store any result')

    face_detector = BlazeFace(args.face_detection_model_path)
    capture = cv2.VideoCapture(args.input)
    if capture.isOpened():
        print(f'Working with {args.input}')
    else:
        print(f'Can not open capture for {args.input}')
        return -1

    total_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    if args.record_video:
        writer = cv2.VideoWriter(str(args.output / 'preview.mp4'),
                                 cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                 capture.get(cv2.CAP_PROP_FPS),
                                 (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if writer.isOpened():
            print('Video will be recorded')
        else:
            print(f'Can not open writer for {args.output / "preview.mp4"}')
            return -1

    if args.save_detections:
        detections_output_dir = args.output / 'detections'
        detections_output_dir.mkdir()
        print(f'Per frame detections result will be written to "{detections_output_dir}"')

    print('Running main loop. Hit ESC on preview to stop or use Ctrl+C in terminal')
    key = -1
    processed_frames = 0
    while key != 27:
        # Capture frame-by-frame
        is_frame_available, frame = capture.read()
        if not is_frame_available:
            print()
            print('No more frames are available')
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = face_detector.detect(rgb_frame)

        transforms = albumentations.Compose([albumentations.augmentations.transforms.ToGray(),
                                            albumentations.augmentations.transforms.Resize(32, 32),
                                            albumentations.augmentations.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
                                            ToTensorV2(),
                                            ])

        left_eye_crop, right_eye_crop = crop_eyes(rgb_frame, detections[0])
        # cv2.imwrite('left_eye.jpg', left_eye_crop)
        # cv2.imwrite('right_eye.jpg', right_eye_crop)
        left_aug = transforms(image=left_eye_crop)
        left_img = left_aug['image'][:1, ...].unsqueeze(0)

        right_aug = transforms(image=right_eye_crop)
        right_img = right_aug['image'][:1, ...].unsqueeze(0)

        device = torch.device("cuda")
        model = Net().to(device)
        model.load_state_dict(torch.load(args.open_closed_weights))
        model.eval()

        left_pred = model(left_img.to(device))

        right_pred = model(right_img.to(device))

        preds = (torch.sigmoid(left_pred), torch.sigmoid(right_pred))

        frame = draw_detections(frame, detections, preds)
        if args.save_detections:
            save_detections(detections_output_dir, processed_frames, detections)


        cv2.namedWindow('Preview', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Preview', frame)
        if args.record_video:
            writer.write(frame)

        key = cv2.waitKey(3)
        processed_frames += 1
        print(f'Processed: {processed_frames} / {total_frames} frames', end='\r')

    print()
    capture.release()
    if args.record_video:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())