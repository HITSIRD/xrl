import os
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


def crop_and_resize(image: np.ndarray, size=256, margin_up=80, margin_down=0) -> np.ndarray:
    """
    裁剪中央方形并缩放图像。
    """
    h, w, _ = image.shape
    min_side = min(h, w)
    top = (h - min_side) // 2
    left = (w - min_side) // 2

    crop_size = margin_up + margin_down
    assert crop_size < min_side
    cropped = image[
        top + margin_up:top + min_side - margin_down, left + crop_size // 2:left + min_side - crop_size // 2]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return resized


def process_dataset(input_root: str, output_root: str, margin_up=80, margin_down=0):
    os.makedirs(output_root, exist_ok=True)

    folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
    print(f"Found {len(folders)} folders.")

    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(input_root, folder)

        # 寻找 .h5 文件
        h5_files = [f for f in os.listdir(folder_path) if f.endswith(".h5")]
        if len(h5_files) != 1:
            print(f"Warning: {folder_path} 中找到 {len(h5_files)} 个 .h5 文件，跳过")
            continue

        h5_path = os.path.join(folder_path, h5_files[0])
        with h5py.File(h5_path, 'r') as f:
            # 假设数据保存在 'images' 或第一个 key 下
            pose = f['pose'][()]
            skill = f['skill'][()]
            q = f['q'][()]
            images = f['rgb'][()]  # shape: (1000, 720, 1280, 3)

        processed = np.stack([crop_and_resize(img, margin_up=margin_up, margin_down=margin_down) for img in images],
                             axis=0)  # shape: (1000, 128, 128, 3)
        # plt.imshow(processed[0])
        # plt.show()

        # 保存
        out_file = os.path.join(output_root, f"{folder}.h5")
        with h5py.File(out_file, 'w') as f_out:
            f_out.create_dataset('pose', data=pose, compression='gzip')
            f_out.create_dataset('q', data=q, compression='gzip')
            f_out.create_dataset('skill', data=skill, compression='gzip')
            f_out.create_dataset('rgb', data=processed, compression='gzip')
        print(f"Saved {out_file}")


def process_h5_skills(input_path, output_path, phase_names, phase_lengths, start_opt):
    os.makedirs(output_path, exist_ok=True)

    files = [f for f in os.listdir(input_path) if f.endswith(".h5")]
    for fname in files:
        with h5py.File(os.path.join(input_path, fname), 'r') as f_in:
            pose = f_in['pose'][()]  # (N,)
            q = f_in['q'][()]  # (N,)
            skill = f_in['skill'][()]  # (N,)
            image = f_in['rgb'][()]  # (N, H, W, 3)

        new_skill, idx = merge_skills(list(skill), phase_names, phase_lengths, start_opt, max_len=50)
        new_skill = np.array(new_skill, dtype='S')  # 转回 byte string

        # 保存新的文件
        with h5py.File(os.path.join(output_path, fname), 'w') as f_out:
            f_out.create_dataset('pose', data=pose[idx], compression='gzip')
            f_out.create_dataset('q', data=q[idx], compression='gzip')
            f_out.create_dataset('skill', data=new_skill, compression='gzip')
            f_out.create_dataset('rgb', data=image[idx], compression='gzip')

        print(f"Processed: {fname}")


def merge_skills(raw_skills, new_skill_names, merge_lengths, start_opt, max_len=1000000):
    assert len(new_skill_names) == len(merge_lengths)

    last_skill = None
    indices = []
    seq_count = [0] * len(new_skill_names)

    for i, skill in enumerate(raw_skills):
        if skill != last_skill:
            print(skill.decode('utf-8'), i)
            last_skill = skill

    sequence = []
    last_skill = None
    idx, merge_idx = 0, 0
    start_idx = len(raw_skills)
    start = False

    for i, skill in enumerate(raw_skills):
        changed = False
        if skill != last_skill:
            last_skill = skill
            changed = True

        if not start:
            flag = skill.decode('utf-8') == start_opt
            start = start | flag
        if start:
            start_idx = min(i, start_idx)
            if changed:
                if idx == merge_lengths[merge_idx]:
                    merge_idx += 1
                    idx = 1
                else:
                    idx += 1

            if seq_count[merge_idx] < max_len:
                sequence.append(new_skill_names[merge_idx])
                indices.append(i)
                seq_count[merge_idx] += 1

    return sequence, np.array(indices)


def preview(path):
    last_skill = None
    with h5py.File(path, 'r') as f:
        plt.imshow(f['rgb'][400])
        plt.show()
        print(f['pose'].shape)
        print(f['skill'].shape)
        print(f['q'].shape)
        print(f['rgb'].shape)

        for i, skill in enumerate(f['skill']):
            # print(skill.decode('utf-8'), i)

            if skill != last_skill:
                print(skill.decode('utf-8'), i)
                last_skill = skill
        # for i in range(len(f['rgb'])):
        #     plt.imsave(f"rgb_seq50/3_{i}.png", f['rgb'][i])


if __name__ == "__main__":
    # 替换为你的路径
    task = 'coffee'
    INPUT_ROOT = f"/home/wenyongyan/下载/dataset/{task}"
    OUTPUT_ROOT = f"/home/wenyongyan/下载/output/{task}"
    output_dir = "/home/wenyongyan/Projects/xrl/src/data/real_kitchen/coffee-50-v0"

    # process_dataset(INPUT_ROOT, OUTPUT_ROOT, margin_up=80)

    # skill_sequence = ['开冰箱门', '靠近芒果', '芒果夹', '芒果抓', '放芒果', '松开', '离开冰箱', '关冰箱门', '开柜门', '靠近草莓jelly' ,
    #                   '草莓jelly抓取', '草莓jelly抓', '放jelly', '松开', '离开柜子']
    # phase_lengths = [1, 6, 1, 1, 6]
    # phase_names = ['open_fridge', 'store_mango', 'close_fridge', 'open_cab', 'store_jello']

    # fruits snacks
    # phase_lengths = [2, 7, 7, 6, 2, 2, 7, 7, 1]
    # phase_names = ['open_fridge', 'store_mango', 'store_lemon', 'store_orange', 'close_fridge', 'open_cabinet', 'store_cheezit', 'store_jello', 'close_cabinet']

    # heat bread
    # phase_lengths = [2, 6, 2, 6, 2, 5, 2]
    # phase_names = ['open_microwave', 'move_bread_to_microwave', 'close_microwave', 'set_time', 'open_microwave',
    #                'move_bread_to_plate', 'close_microwave']

    # cola
    # phase_lengths = [2, 6, 2, 7, 6, 5, 2, 6, 2]
    # phase_names = ['open(fridge)', 'move(ice_cup, table)', 'close(fridge)', 'pour(cola, ice_cup)', 'move(straw, ice_cup)',
    #                'move(cola, bin)', 'open(fridge)', 'move(milk, fridge)', 'close(fridge)']
    #

    # coffee
    phase_lengths = [2, 7, 2, 7, 2, 6, 6, 7, 7, 2, 7]
    phase_names = ['move(funnel, pot)', 'pour_preheat(gooseneck_kettle, funnel)', 'move(funnel, table)',
                   'pour(pot, cup)',
                   'move(funnel, pot)', 'pour(coffee_powder, funnel)', 'pour(kettle, gooseneck_kettle)',
                   'pour_preheat(gooseneck_kettle, funnel)', 'pour(gooseneck_kettle, funnel)', 'move(funnel, table)',
                   'pour(pot, coffee_cup)']

    process_h5_skills(OUTPUT_ROOT, output_dir, phase_names, phase_lengths, 'reset_funnel')
    # for i in range(10):
    #     preview(os.path.join(output_dir, f"{task}_{i}.h5"))
    # for i in range(10):
    #     preview(os.path.join(OUTPUT_ROOT, f"{task}_{i}.h5"))
