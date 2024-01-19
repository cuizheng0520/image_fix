import glob
import os
import time

import numpy as np
import paddlehub as hub
import cv2
import matplotlib.pyplot as plt


class WatermarksRemover:
    def __init__(self, words='', image_save_folder='data/results', show_process=True):
        self.ocr = hub.Module(name="chinese_ocr_db_crnn_server")
        self.use_gpu = False
        self.words = words
        self.image_save_folder = image_save_folder
        self.show_process = show_process

    @staticmethod
    def read_image(image_path):
        return cv2.imread(image_path)

    def repair(self, image_path, expand=True, expand_l=10, expand_r=10, expand_t=10,
               expand_b=10):
        st = time.time()
        image = self.read_image(image_path)
        marked_image = image.copy()
        mask = np.zeros(image.shape[:2], np.uint8)

        results = self.ocr.recognize_text(
            images=[image],
            use_gpu=self.use_gpu,
            output_dir='ocr_result',
            visualization=False,
            box_thresh=0.5,
            text_thresh=0.5
        )

        for result in results:
            for info in result['data']:
                if info['text'] == self.words:
                    text_box_position = np.array(info['text_box_position'])
                    if expand:
                        self.adjust_text_box(text_box_position, expand_l, expand_r, expand_t, expand_b)
                    self.add_to_mask(mask, text_box_position)
                    self.draw_text_box(marked_image, text_box_position, info)

        radius = 5
        result_image = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
        # image save
        self.save_images(image_path, result_image, marked_image)
        et = time.time()
        return round(et - st, 4)

    @staticmethod
    def adjust_text_box(text_box_position, expand_l, expand_r, expand_t, expand_b):
        # 水平方向调整
        text_box_position[1][0] += expand_r  # 右上角
        text_box_position[2][0] += expand_r  # 右下角
        text_box_position[0][0] -= expand_l  # 左上角
        text_box_position[3][0] -= expand_l  # 左下角
        # 垂直方向调整
        text_box_position[0][1] -= expand_t  # 左上角
        text_box_position[1][1] -= expand_t  # 右上角
        text_box_position[2][1] += expand_b  # 右下角
        text_box_position[3][1] += expand_b  # 左下角

    @staticmethod
    def add_to_mask(mask, text_box_position):
        cv2.fillPoly(mask, [text_box_position], (255, 255, 255))

    @staticmethod
    def draw_text_box(image, text_box_position, info):
        cv2.polylines(image, [text_box_position], isClosed=True, color=(0, 255, 0), thickness=3)
        x, y = text_box_position[0]
        text_to_display = f"{info['text']}, Confidence: {info['confidence']:.2f}"
        text_size = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # 上移10个像素
        offset = 15
        y -= offset  # 更新y坐标

        rect_x1 = x - 10
        rect_y1 = y - text_size[1] - 20
        rect_x2 = x + text_size[0] + 10
        rect_y2 = y + 10

        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.putText(image, text_to_display, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def save_images(self, original_image_path, result_image, marked_image):
        folder = self.image_save_folder

        if not os.path.exists(folder):
            os.makedirs(folder)

        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        original_image_filename = f"{base_name}_1_before.jpg"
        marked_image_filename = f"{base_name}_2_marked.jpg"
        result_image_filename = f"{base_name}_3_after.jpg"

        if self.show_process:
            cv2.imwrite(os.path.join(folder, original_image_filename), self.read_image(original_image_path))
            cv2.imwrite(os.path.join(folder, marked_image_filename), marked_image)
        cv2.imwrite(os.path.join(folder, result_image_filename), result_image)

    def save_process_image(self):
        if not self.show_process:
            return
        if not os.path.exists(f"{self.image_save_folder}/process"):
            os.makedirs(f"{self.image_save_folder}/process")
        image_names = self.get_image_filenames()
        num_images = len(image_names)
        num_rows = (num_images + 2) // 3
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        if num_rows == 1:
            axes = [axes]
        idx = 0
        for row in axes:
            for ax in row:
                if idx < num_images:
                    image_name = image_names[idx]
                    image = cv2.imread(image_name)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image_rgb)
                    ax.set_title(image_name)
                    idx += 1
        plt.tight_layout()
        plt.savefig(f"{self.image_save_folder}/process/process_log.png")
        plt.show()

    def get_image_filenames(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
        image_files = []
        for extension in image_extensions:
            image_files.extend(glob.glob(f"{self.image_save_folder}/{extension}"))
        return sorted(image_files)
