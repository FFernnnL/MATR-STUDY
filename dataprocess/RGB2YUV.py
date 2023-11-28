import os
import cv2


def rgb_to_yuv(rgb_img):
    yuv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv_img)
    return y, u, v


def save_component(component, folder, filename):
    save_path = os.path.join(folder, filename)
    cv2.imwrite(save_path, component)


def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(input_folder, filename)
            rgb_img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if rgb_img is not None:
                y, u, v = rgb_to_yuv(rgb_img)

                save_component(y, os.path.join(output_folder, 'Y'), filename)
                save_component(u, os.path.join(output_folder, 'U'), filename)
                save_component(v, os.path.join(output_folder, 'V'), filename)


if __name__ == "__main__":
    input_folder = "path/to/your RGB image directory"  # 输入RGB图像文件夹路径
    output_folder = "path/to/your YUV image directory"  # 输出YUV的文件夹路径
    process_images(input_folder, output_folder)
