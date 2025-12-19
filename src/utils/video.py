import io
import os

import h5py
import numpy as np
from PIL import Image
from torchvision.transforms import Resize
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
import os
import textwrap


def ch_first2last(video):
    return video.transpose((0, 2, 3, 1))


def ch_last2first(video):
    return video.transpose((0, 3, 1, 2))


def resize_video(video, size):
    if video.shape[1] == 3:
        video = np.transpose(video, (0, 2, 3, 1))
    transformed_video = np.stack([np.asarray(Resize(size)(Image.fromarray(im))) for im in video], axis=0)
    return transformed_video


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(video_frames, filename, fps=60, video_format='mp4'):
    assert fps == int(fps), fps
    import skvideo.io
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p',
            # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )


def create_video_grid(col_and_row_frames):
    video_grid_frames = np.concatenate([
        np.concatenate(row_frames, axis=-2)
        for row_frames in col_and_row_frames
    ], axis=-3)

    return video_grid_frames


def create_video_from_dataset(
        path="./experiments/",
        output_dir="./data/real_kitchen/",
        output_video="output.mp4",
        fps=10
):
    with h5py.File(path, 'r') as h5_file:
        images = h5_file['rgb'][()]

    frames = []
    print(images.shape)

    # 将每个图像保存为临时帧
    for i, image in enumerate(images):
        # 如果图像是 numpy 数组，转换为 PIL Image
        if isinstance(image, np.ndarray):
            # 处理不同维度的图像数据
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW 格式
                if image.shape[0] == 1:
                    image = image.squeeze(0)  # 转换为 HW
                else:
                    image = np.transpose(image, (1, 2, 0))  # 转换为 HWC

            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        frame_array = np.array(pil_image)
        frames.append(frame_array)
        if i % 100 == 0:
            print(f'frame {i}')
        if i % 100 == 0:
            print(f'frame {i}')

    # 创建视频剪辑
    clip = mp.ImageSequenceClip(frames, fps=fps)

    # 生成视频文件
    output_path = os.path.join(output_dir, output_video)
    clip.write_videofile(output_path, codec='libx265', bitrate='2000k',
                         ffmpeg_params=['-tag:v', 'hvc1', '-pix_fmt', 'yuv420p'])

    # 清理临时帧文件
    for frame in frames:
        if os.path.exists(frame):
            os.remove(frame)

    print(f"视频已生成：{output_path}")


def create_video_from_pdfs_and_markdowns(
        pdf_dir="./experiments/",
        md_dir="./experiments/",
        output_dir="./experiments/",
        output_video="output.mp4",
        num_files=28,
        frame_duration=2,
        fps=30,
        font_path="NotoSansCJK-Regular.ttc",
        font_size=36,
        text_height=250,
        text_width=64,
        language_output=True,
        clean_tmp=True
):
    """
    从 PDF 文件（图片）和 Markdown 文件（文字描述）生成视频。

    参数:
        pdf_dir (str): PDF 文件所在目录
        md_dir (str): Markdown 文件所在目录
        output_video (str): 输出视频文件名
        num_files (int): 文件数量
        frame_duration (int): 每帧显示时间（秒）
        fps (int): 视频帧率
        font_path (str): 字体文件路径
        font_size (int): 文字大小
        text_height (int): 文字区域高度
        text_width (int): 每行文字最大宽度（字符数）
        language_putput (bool): 是否显示自然语言解释
        clean_up (bool): 是否清理临时文件

    返回:
        None，生成视频文件
    """
    # 创建临时帧存储目录
    os.makedirs(output_dir, exist_ok=True)

    # 存储所有帧路径
    frames = []

    # try:
    # 遍历 PDF 和 Markdown 文件
    for i in range(num_files):
        pdf_path = os.path.join(pdf_dir, f"skill_influence_{i}.pdf")
        md_path = os.path.join(md_dir, f"explanation_{i}.md")

        # 提取 PDF 中的图片
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[0]  # 假设每 PDF 只有一页
        zoom = 2.0  # 你可以尝试 2.5 或 3.0 视清晰度需求而定
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_doc.close()

        if language_output:
            # 读取 Markdown 文件中的文字
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                # text = text.replace("**", "")
                # text = text.replace(" **", "")
                # text = text.replace("** ", "")
                # text = text.replace("### ", "")

        # 创建新图片：上半部分图片 + 下半部分文字
        img_width, img_height = img.size
        if language_output:
            new_height = img_height + text_height
        else:
            new_height = img_height
        new_img = Image.new("RGB", (img_width, new_height), color="white")
        new_img.paste(img, (0, 0))

        if language_output:
            # 绘制文字
            draw = ImageDraw.Draw(new_img)
            font = ImageFont.truetype(font_path, font_size)
            wrapped_text = textwrap.fill(text, width=text_width)
            draw.text((10, img_height + 10), wrapped_text, font=font, fill="black")

        # 保存帧图片
        frame_path = os.path.join(output_dir, f"tmp_frame_{i}.png")
        new_img.save(frame_path)
        frames.append(frame_path)

    # 合成视频
    clip = mp.ImageSequenceClip(frames, durations=[frame_duration] * len(frames))
    clip.fps = fps
    clip.write_videofile(os.path.join(output_dir, output_video), codec="libx264")

    print(f"视频已生成：{output_video}")

    if clean_tmp:
        for frame in frames:
            if os.path.exists(frame):
                os.remove(frame)

    # except Exception as e:
    #     print(f"发生错误：{str(e)}")
    # finally:
    #     # 清理临时帧文件（可选）
    #     for frame in frames:
    #         if os.path.exists(frame):
    #             os.remove(frame)


# domain = 'cola'
# episode = 3
# create_video_from_dataset(path=f'/home/wenyongyan/下载/dataset/{domain}/{domain}_{episode}/traj.h5',
#                           output_video=f'{domain}_{episode}.mp4')
