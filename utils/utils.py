# encoding=utf-8

import os
import shutil
# from PIL import Image, ImageDraw, ImageFont
from collections import Counter

import cv2
import numpy as np
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET


def find_most_free_gpu():
    """
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp.py')
    memory_left_gpu = [int(x.split()[2]) for x in open('tmp.py', 'r').readlines()]

    most_free_gpu_idx = np.argmax(memory_left_gpu)
    # print(str(most_free_gpu_idx))
    return int(most_free_gpu_idx)


def select_device(device='', apex=False, batch_size=None):
    """
    :param device:
    :param apex:
    :param batch_size:
    :return:
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def parse_darknet_cfg(path):
    """
    :param path:
    :return:
    """
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')

    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    m_defs = []  # module definitions

    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            m_defs.append({})
            m_defs[-1]['type'] = line[1:-1].rstrip()
            if m_defs[-1]['type'] == 'convolutional':
                m_defs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return np-array
                m_defs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                m_defs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    m_defs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    m_defs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride',
                 'pad', 'activation', 'layers', 'groups', 'from',
                 'mask', 'anchors', 'classes', 'num', 'jitter',
                 'ignore_thresh', 'truth_thresh', 'random', 'stride_x', 'stride_y',
                 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'groups',
                 'group_id', 'probability']

    f = []  # fields
    for x in m_defs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields

    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return m_defs


def load_darknet_weights(model, weights, cutoff=0):
    """
    :param model:
    :param weights:
    :param cutoff:
    :return:
    """
    print('Cutoff: ', cutoff)

    # ----- Parses and loads the weights stored in 'weights'

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        model.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        model.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    # for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
    for i, (mdef, module) in enumerate(zip(model.module_defs, model.module_list)):
        if cutoff != 0 and i > cutoff:
            break

        if mdef['type'] == 'convolutional' or mdef['type'] == 'deconvolutional':  # how to load 'deconvolutional' layer
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases

                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb

                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb

                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb

                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)

                conv.bias.data.copy_(conv_b)
                ptr += nb

            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def ivt_tensor_to_img(tensor,
                      mean=np.array([0.485, 0.456, 0.406]),
                      std=np.array([0.229, 0.224, 0.225])):
    """
    ivt tensor to img(numpy array)
    :param tensor:
    :param mean:
    :param std:
    :return:
    """
    # CHW -> HWC
    tensor = tensor.numpy().transpose((1, 2, 0))

    # De-standardization
    tensor = tensor * std + mean

    # Clamping
    output = np.clip(tensor, 0, 1)

    return output


def tensor_to_img(tensor,
                  mean=np.array([0.5, 0.5, 0.5]),
                  std=np.array([0.5, 0.5, 0.5]),
                  invt_channels=True):
    """
    :param tensor: CPU Tensor
    :param mean: Numpy Image
    :param std:
    :param invt_channels:
    :return:
    """
    tensor = ivt_tensor_to_img(tensor, mean, std)
    tensor = tensor * 255.0  # [0, 1] _> [0, 255]
    tensor = tensor.astype(np.uint8)

    if invt_channels:
        tensor = tensor[:, :, ::-1]  # RGB ——> BGR
    return tensor


def find_files_of_ext(root, f_list, ext):
    """
    Find all files with an extension
    """
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(ext):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            find_files_of_ext(f_path, f_list, ext)


def gen_txt_list_of_dirs(roots_list,
                         out_txt_path,
                         ext=".jpg",
                         ratio=1.0):
    """
    Args:
        roots_list:
        out_txt_path:
        ext:
        ratio:
    Returns:
    """
    img_path_list = []
    for dir_path in roots_list:
        if not os.path.isdir(dir_path):
            print("[Warning]: invalid dir path.")
            continue

        img_paths = [dir_path + "/" + x for x in os.listdir(dir_path) if x.endswith(ext)]
        img_path_list.extend(img_paths)
        print("{:d} img paths found.".format(len(img_path_list)))

    cnt = 0
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for img_path in tqdm(img_path_list):
            if not os.path.isfile(img_path):
                print("[Warning]: invalid img path: {:s}.".format(img_path))
                continue

            if np.random.random() <= ratio:
                f.write(img_path + "\n")
                cnt += 1
    print("Total {:d} img paths written to txt list.".format(cnt))


def gen_txt_list_of_dir(in_root,
                        out_txt_path,
                        ext=".jpg",
                        min_size_thresh=0,
                        ratio=1.0):
    """
    """
    if not os.path.isdir(in_root):
        print("[Err]: invalid root dir path {:s}".format(in_root))
        return

    print("[Info]: Finding {:s} files in dir {:s}...".format(ext, in_root))
    f_list = []
    find_files_of_ext(in_root, f_list, ext)
    print("[Info]: Total {:d} samples({:s}) found.".format(len(f_list), ext))

    print("Start filtering files...")
    cnt = 0
    out_txt_path = os.path.abspath(out_txt_path)
    with tqdm(total=len(f_list)) as progress_bar:
        with open(out_txt_path, "w", encoding="utf-8") as f:
            for img_path in tqdm(f_list):
                if not os.path.isfile(img_path):
                    print("[Warning]: invalid img path: {:s}.".format(img_path))
                    continue

                if min_size_thresh > 0.0:
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    H, W = img.shape[:2]  # HWC or HW
                    if H < min_size_thresh or W < min_size_thresh:
                        print("[Info]: Skip {:s}, for min size limit.".format(img_path))
                        continue

                if np.random.random() <= ratio:
                    f.write(img_path + "\n")
                    cnt += 1

                progress_bar.update()

    print("Total {:d} img paths written to txt list.".format(cnt))


def mv_files(txt_path, dst_dir, ratio=1.0):
    """
    Moving(coping) files from TXTlist to a dir.
    """
    if not os.path.isfile(txt_path):
        print("[Err]: invalid txt file path.")
        return

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        print("{:s} made.".format(dst_dir))

    cnt = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            f_path = line.strip()

            if not os.path.isfile(f_path):
                print("[Warning]: invalid file path: {:s}".format(f_path))
                continue

            if np.random.random() <= ratio:
                f_name = os.path.split(f_path)[-1]
                dst_path = dst_dir + "/" + f_name
                if not os.path.isfile(dst_path):
                    shutil.copy(f_path, dst_dir)

                cnt += 1
                if (i + 1) % 100 == 0:
                    print("Total {:d} files transferred.".format(cnt))


def gen_csv(dir_list,
            csv_path,
            ext=".bmp",
            data_mode="syn",
            write_mode="w"):
    """
    Generate CSV file from dir list
    :param data_mode: syn | ref | ugc
    :param write_mode: w | a+
    """
    if len(dir_list) == 0:
        print("[Err]: empty dir list.")
        return

    print("Data mode: ", data_mode)
    print("Write mode: ", write_mode)

    N_DISTORTIONS = 25
    N_CLS = N_DISTORTIONS * 5 + 1
    print("N_CLS: {:d}".format(N_CLS))
    cnt = 0
    with open(csv_path, write_mode, encoding="utf-8") as f:
        if write_mode != "a+":
            f.write(",Unnamed: 0,File_names,labels\n")
        for dir_path in dir_list:
            print("Processing {:s}...".format(dir_path))
            img_names = [x for x in os.listdir(dir_path) if x.endswith(ext)]
            for img_name in img_names:
                img_path = dir_path + "/" + img_name
                if not os.path.isfile(img_path):
                    continue

                print("processing {:s}".format(img_path))

                items = img_name.split(".")[0].split("_")
                # print(items)

                if len(items) > 3 and data_mode == "syn":
                    distort_type = int(items[-2])
                    distort_level = int(items[-1])

                f.write('{:d},{:d},{:s},"['.format(cnt, cnt, img_path))

                for i in range(N_CLS):
                    if data_mode == "syn":
                        if i == (distort_type - 1) * 5 + (distort_level - 1) + 1:
                            if i == N_CLS - 1:
                                f.write("1.0")
                            else:
                                f.write("1.0, ")
                        elif 0 <= i < N_CLS - 1:
                            f.write("0.0, ")
                        elif i == N_CLS - 1:
                            f.write("0.0")
                    elif data_mode == "ref":
                        if i == 0:
                            f.write("1.0, ")
                        elif 0 <= i < N_CLS - 1:
                            f.write("0.0, ")
                        elif i == N_CLS - 1:
                            f.write("0.0")
                    elif data_mode == "ugc":
                        if 0 <= i < N_CLS - 1:
                            f.write("0.0, ")
                        elif i == N_CLS - 1:
                            f.write("0.0")
                    else:
                        print("[Err]: invalid data mode, data mode should be:"
                              "syn | ref | ugc")
                f.write(']"\n')

                cnt += 1


def clear_dirs(root, min_limit=2, max_limit=30):
    """
    Args:
        root:
        min_limit:
        max_limit:
    Returns:
    """
    if not os.path.isdir(root):
        print("\n[Err]: invalid root dir: {:s}.\n".format(root))
        return

    sub_dirs = [root + "/" + x for x in os.listdir(root) if os.path.isdir(root + "/" + x)]
    with tqdm(total=len(sub_dirs)) as progress_bar:
        for dir_path in sub_dirs:
            dir_name = os.path.split(dir_path)[-1]
            if len(dir_name) < 7:
                shutil.rmtree(dir_path)
                print("\n{:s} removed.\n".format(dir_path))

            f_list = [x for x in os.listdir(dir_path)]
            if len(f_list) > max_limit or len(f_list) < min_limit:
                shutil.rmtree(dir_path)
                print("\n{:s} removed.\n".format(dir_path))
            progress_bar.update()


def gen_train_set(train_list):
    """
    Generate training set
    """
    dir_list = [train_list[0]]
    gen_csv(dir_list=dir_list,
            csv_path="../csv_files/plates_syn.csv",
            ext=".bmp",
            data_mode="syn",
            write_mode="w")

    dir_list = [train_list[1]]
    gen_csv(dir_list=dir_list,
            csv_path="../csv_files/plates_syn.csv",
            ext=".jpg",
            data_mode="ref",
            write_mode="a+")

    dir_list = train_list[2:]
    gen_csv(dir_list=dir_list,
            csv_path="../csv_files/plates_ugc.csv",
            ext=".jpg",
            data_mode="ugc",
            write_mode="w")


def parse_xml_for_plate(jpg_path,
                        xml_path,
                        dst_dir,
                        counter,
                        ext=".jpg",
                        clock_wise=True,
                        split_dir=False,
                        logging=True):
    """
    Args:
        jpg_path:
        xml_path:
        dst_dir:
        counter:
        ext,
        clock_wise:
        split_dir:
        logging:
    Returns:
    """
    assert os.path.isdir(dst_dir)

    if not os.path.isfile(jpg_path):
        print("[Err]: invalid jpg path: {:s}.".format(os.path.abspath(jpg_path)))
        return

    if not os.path.isfile(xml_path):
        print("\n[Warning]: invalid xml path {:s}.\n".format(os.path.abspath(xml_path)))
        return

    ## --------- Read in image
    img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)

    ## ---------- parse xml
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_info = f.read()

        try:
            root = ET.fromstring(xml_info)
        except Exception as e:
            print("[Err]: can not parse xml file: {:s}".format(xml_path))

        # obj = root.find('markNode')
        # if obj is None:
        #     print("[Err]: find markNode failed!")
        #     return

        for obj in root.iter('object'):  # 遍历object, 每个object代表一个标注目标
            target_type = obj.find('targettype')
            target_name = target_type.text
            if target_name != "plate":  # 读取车牌对象
                continue

            ## ----- 解析车牌属性
            color_node = obj.find("color")
            if color_node is None:
                print("[Warning]: can not find color!")
                continue
            plate_color = color_node.text

            type_ndoe = obj.find("mode")
            if type_ndoe is None:
                print("[Warning]: can not find type!")
                continue
            plate_type = type_ndoe.text

            ## ----- 解析车牌号码
            char_nodes = obj.find("characters")
            if char_nodes is None:
                print("[Warning]: can not find character!")
                continue

            plate_number = []
            for char in char_nodes.iter("char"):
                char = char.find("data").text
                plate_number.append(char)
            n_chars = len(plate_number)
            plate_number = "".join(plate_number)

            ## ----- 解析4个角点: 左上角逆时针
            vertex_nodes = obj.find("vertexs")
            if vertex_nodes is None:
                print("[Warning]: can not find vertex!")
                continue

            vertices = []
            for vertex in vertex_nodes.iter("vertex"):
                x = int(vertex.find("x").text)
                y = int(vertex.find("y").text)
                vertices.append([x, y])
            assert len(vertices) == 4

            # draw4Corners(img, vertices)
            # cv2.imwrite(vis_path, img)
            # print("{:s} written.".format(vis_path))

            # 基于4个角点Warp到正视图
            if len(vertices) != 4:
                print("[Warning]: corners number is not 4!.")
                continue

            vertex_arr = np.array(vertices)
            pt_ord = [min(vertex_arr[:, 0]), min(vertex_arr[:, 1])]
            pt_ord[0] = pt_ord[0] if pt_ord[0] >= 0 else 0
            pt_ord[1] = pt_ord[1] if pt_ord[1] >= 0 else 0

            rect_w = max(vertex_arr[:, 0]) - min(vertex_arr[:, 0])
            rect_h = max(vertex_arr[:, 1]) - min(vertex_arr[:, 1])
            if rect_w <= 0 or rect_h <= 0:
                print("[Warning]: invalid Rect, rect_w: {:d}, rect_h: {:d}"
                      .format(rect_w, rect_h))
                continue

            rect = img[pt_ord[1]: pt_ord[1] + rect_h, pt_ord[0]: pt_ord[0] + rect_w, :]
            h, w = rect.shape[:2]
            if h == 0 or w == 0:
                print("[Warning]: invalid Rect of plate!")
                continue

            src_pts, dst_pts = [], []

            if plate_type == "单":
                for i, vertex in enumerate(vertex_arr):
                    src_pts.append([vertex[0] - pt_ord[0], vertex[1] - pt_ord[1]])  # 转换到车牌局部坐标系

                    if clock_wise:  # 顺时针标注的4角点
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([191, 0])
                        elif i == 2:
                            dst_pts.append([191, 63])
                        elif i == 3:
                            dst_pts.append([0, 63])

                    else:  # 逆时针标注的4角点
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([0, 63])
                        elif i == 2:
                            dst_pts.append([191, 63])
                        elif i == 3:
                            dst_pts.append([191, 0])

            elif plate_type == "双":
                for i, vertex in enumerate(vertex_arr):
                    src_pts.append([vertex[0] - pt_ord[0], vertex[1] - pt_ord[1]])  # 转换到车牌局部坐标系

                    if clock_wise:
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([191, 0])
                        elif i == 2:
                            dst_pts.append([191, 95])
                        elif i == 3:
                            dst_pts.append([0, 95])

                    else:
                        if i == 0:
                            dst_pts.append([0, 0])
                        elif i == 1:
                            dst_pts.append([0, 95])
                        elif i == 2:
                            dst_pts.append([191, 95])
                        elif i == 3:
                            dst_pts.append([191, 0])

            if not (len(src_pts) == 4 and len(dst_pts)) == 4:
                print("[Warning]: invalid src_pts or dst_pts!")
                continue

            src_pts, dst_pts = np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)
            mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

            if plate_type == "单":
                try:
                    plate_img_warped = cv2.warpPerspective(rect, mat, (192, 64))
                except Exception as e:
                    print(e)
                    continue
            elif plate_type == "双":
                try:
                    plate_img_warped = cv2.warpPerspective(rect, mat, (192, 96))
                except Exception as e:
                    print(e)
                    continue

            ## ----- save warped image
            if plate_type == "单":
                plate_type = "single"
            elif plate_type == "双":
                plate_type = "double"

            if split_dir:
                save_dir = dst_dir + "/" + plate_number
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
            else:
                save_dir = dst_dir

            pl_number = plate_number.replace("*", "_")  # .replace("~", ".")
            save_path = save_dir + "/" \
                        + pl_number \
                        + "_" + plate_color \
                        + "_" + plate_type \
                        + "_" + str(counter["id"]) + ext
            if not os.path.isfile(save_path):
                # cv2.imwrite(save_path, plate_img_warped)
                cv2.imencode(ext, plate_img_warped)[1].tofile(save_path)

            if logging:
                print("\n")
                print("Rect width: ", rect_w)
                print("Rect height: ", rect_h)
                print("车牌类型: ", plate_type)
                print("车牌颜色: ", plate_color)
                print("车牌号码: {:s}, {:d}字符".format(plate_number, n_chars))
                print("\n")

            ## ----- sample counter updating
            counter.update(["id"])


def parse_plates(src_dir,
                 dst_dir,
                 clock_wise=True,
                 ext=".jpg",
                 split_dir=False,
                 logging=True):
    """
    Args:
        src_dir:
        dst_dir:
        clock_wise: True: clock_wise | False: anti-clock_wise
        ext:
        split_dir:
    Returns:
    """
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir.")
        return

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        print("{:s} made.".format(dst_dir))

    img_list = []
    find_files_of_ext(src_dir, img_list, ext=".jpg")
    img_list.sort()
    xml_list = [x.replace(".jpg", ".xml") for x in img_list]

    item_list = [(img_path, xml_path) for img_path, xml_path in zip(img_list, xml_list)]
    print("[Info]: Total {:d} valid items found.".format(len(item_list)))

    counter = Counter({"id": 0})
    with tqdm(total=len(item_list)) as progress_bar:
        for i, (img_path, xml_path) in enumerate(item_list):
            ## ----- Parsing & Warping
            parse_xml_for_plate(img_path, xml_path, dst_dir,
                                counter, ext, clock_wise, split_dir, logging)
            ## -----

            progress_bar.update()

    # for i, (img_path, xml_path) in enumerate(item_list):
    #     ## ----- Parsing & Warping
    #     parseXmlForPlate(img_path, xml_path, dst_dir, counter, ext, split_dir=False)
    #     ## -----
    #     if (i + 1) % 300 == 0:
    #         print("{:.3f}% completed.".format((i + 1) / len(item_list)  * 100.0))


if __name__ == "__main__":
    # gen_txt_list_of_dir(in_root="/users/zhoukai/data/Plate_char_test1225",
    #                 out_txt_path="../data/test.txt",
    #                 min_size_thresh=0)

    # mv_files(txt_path="../data/train.txt",
    #         dst_dir="/mnt/diskd/MCDataset/train",
    #         ratio=1.0)

    dir_list = [
        "/mnt/diskd/dist_plates",
        "/mnt/diskd/ref_plates",
        "/mnt/diskd/even/LPDataRealBase",
        "/mnt/diskd/even/LPDataDouble"
    ]
    #
    # gen_csv(dir_list=dir_list,
    #        csv_path="../csv_files/plates_ugc.csv",
    #        ext=".jpg",
    #        data_mode="ugc",
    #        write_mode="w")

    # gen_train_set(dir_list)

    # parse_plates(src_dir="/users/sunshangyun/data/plate_first_data/mainland",
    #              dst_dir="/mnt/diskd/even/plates",
    #              clock_wise=True,
    #              ext=".jpg",
    #              split_dir=True,
    #              logging=False)
    clear_dirs(root="/mnt/diskd/even/plates", min_limit=6, max_limit=25)

    print("Done.")
