import copy
import argparse
import cv2
import numpy as np
import onnxruntime

class PP_MattingV2:
    def __init__(self, modelpath, background_url, conf_thres=0.65):
        self.conf_threshold = conf_thres
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath)
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        self.input_shape = self.onnx_session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.background_image = self.load_and_prepare_background(background_url)

    def prepare_input(self, image):
        input_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=(self.input_width, self.input_height))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def load_and_prepare_background(self, background_url):
        background_image = cv2.imread(background_url)
        return self.crop_background(background_image)

    def detect(self, image):
        input_image = self.prepare_input(image)

        # Perform inference on the image
        result = self.onnx_session.run([self.output_name], {self.input_name: input_image})

        # Post process: squeeze
        segmentation_map = result[0]
        segmentation_map = np.squeeze(segmentation_map)

        image_width, image_height = image.shape[1], image.shape[0]
        dst_image = copy.deepcopy(image)
        segmentation_map = cv2.resize(
            segmentation_map,
            dsize=(image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = np.where(segmentation_map > self.conf_threshold, 1, 0)
        mask = np.stack((mask,) * 3, axis=-1).astype('uint8')
        human_parts = np.where(mask == 1, image, 0)
        background_parts = np.where(mask == 0, self.background_image, 0)
        return cv2.add(human_parts, background_parts)

    def crop_background(self, background_image):
        target_height, target_width = 480, 640
        height, width = background_image.shape[:2]
        if height >= width:
            crop_height = int(width * target_height / target_width)
            start_y = (height - crop_height) // 2
            cropped = background_image[start_y:start_y + crop_height, :]
        else:
            crop_width = int(height * target_width / target_height)
            start_x = (width - crop_width) // 2
            cropped = background_image[:, start_x:start_x + crop_width]
        return cv2.resize(cropped, (target_width, target_height))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default="image path")
    parser.add_argument('--modelpath', type=str, default="model path")
    parser.add_argument('--background_url', type=str,  default="background image path")
    parser.add_argument('--confThreshold', default=0.65, type=float, help='class confidence')
    parser.add_argument('--use_video', type=int, default=0, help="0 for image, 1 for local video, 2 for webcam")
    parser.add_argument('--videopath', type=str, help="video path")
    parser.add_argument('--save_imgpath', type=str, help="path to save the resulting image")
    parser.add_argument('--save_videopath', type=str, help="path to save the resulting video")
    args = parser.parse_args()

    segmentor = PP_MattingV2(args.modelpath, background_url=args.background_url, conf_thres=args.confThreshold)

    if args.use_video == 0:
        # 进行图片推理
        srcimg = cv2.imread(args.imgpath)

        # Detect Objects
        dstimg = segmentor.detect(srcimg)

        # Save the resulting image
        if args.save_imgpath:
            cv2.imwrite(args.save_imgpath, dstimg)

    elif args.use_video == 1:
        # 进行本地视频推理
        cap = cv2.VideoCapture(args.videopath)
        # Get video writer initialized to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save_videopath, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dstimg = segmentor.detect(frame)

            # Write the frame into the file
            out.write(dstimg)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            cv2.imshow('PP-MattingV2 in ONNXRuntime', dstimg)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    elif args.use_video == 2:
        # 进行摄像头视频推理
        cap = cv2.VideoCapture(0)  # 打开摄像头
        # Get video writer initialized to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save_videopath, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dstimg = segmentor.detect(frame)

            # Write the frame into the file
            out.write(dstimg)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            cv2.imshow('PP-MattingV2 in ONNXRuntime', dstimg)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


#python change.py --modelpath models/ppmattingv2_stdc1_human_736x1280.onnx --use_video 2 --save_videopath outputs/output1.mp4 --background_url background/home.png