import os
import cv2


def yuv_to_rgb(y, u, v):
    yuv_img = cv2.merge([y, u, v])
    rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
    return rgb_img


def load_component(folder, filename):
    file_path = os.path.join(folder, filename)
    component = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return component


def process_images(y_folder, u_folder, v_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(y_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            y = load_component(y_folder, filename)
            u = load_component(u_folder, filename)
            v = load_component(v_folder, filename)

            if y is not None and u is not None and v is not None:
                rgb_img = yuv_to_rgb(y, u, v)
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, rgb_img)


if __name__ == "__main__":
    y_folder = "path/to/your fusion result"  # 将模型融合结果作为Y分量
    u_folder = "path/to/your TEST image/SPECT/YUV/U"  # 测试用SPECT图像的U分量
    v_folder = "path/to/your TEST image/SPECT/YUV/V"  # 测试用SPECT图像的V分量
    output_folder = "path/to/your TRUE RESULT"  # 输出文件夹路径

    process_images(y_folder, u_folder, v_folder, output_folder)
