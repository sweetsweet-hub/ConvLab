import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy import metric
from outputs_nii import parser


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0


def process_label(label):
    RV = label == 1
    Myo = label == 2
    LV = label == 3

    return RV, Myo, LV


def main():
    args = parser.parse_args()
    args.labelsdir = args.data_dir + "labelsTs/"
    label_list = sorted(glob.glob(os.path.join(args.labelsdir, '*nii.gz')))
    output_list = sorted(glob.glob(os.path.join(args.output_directory, '*nii.gz')))
    print("loading success...")
    print("label_list: \n", label_list)
    print("output_list: \n", output_list)
    Dice_RV = []
    Dice_Myo = []
    Dice_LV = []

    hd_RV = []
    hd_Myo = []
    hd_LV = []

    fw = open(os.path.join(args.output_directory) + '/dice_hd_result.txt', 'a')

    for label_path, infer_path in zip(label_list, output_list):
        print(label_path.split('/')[-1], " and ", infer_path.split('/')[-1])  # win系统用"\\"
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_RV, label_Myo, label_LV = process_label(label)
        infer_RV, infer_Myo, infer_LV = process_label(infer)

        Dice_RV.append(dice(infer_RV, label_RV))
        Dice_Myo.append(dice(infer_Myo, label_Myo))
        Dice_LV.append(dice(infer_LV, label_LV))

        hd_RV.append(hd(infer_RV, label_RV))
        hd_Myo.append(hd(infer_Myo, label_Myo))
        hd_LV.append(hd(infer_LV, label_LV))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_RV: {:.4f}\n'.format(Dice_RV[-1]))
        fw.write('Dice_Myo: {:.4f}\n'.format(Dice_Myo[-1]))
        fw.write('Dice_LV: {:.4f}\n'.format(Dice_LV[-1]))

        fw.write('hd_RV: {:.4f}\n'.format(hd_RV[-1]))
        fw.write('hd_Myo: {:.4f}\n'.format(hd_Myo[-1]))
        fw.write('hd_LV: {:.4f}\n'.format(hd_LV[-1]))

        dsc = []  # 每次初始化，用于求当前样本的均值
        HD = []  # 每次初始化，用于求当前样本的均值
        dsc.append(Dice_RV[-1])
        dsc.append((Dice_Myo[-1]))
        dsc.append(Dice_LV[-1])
        fw.write('DSC:' + str(np.mean(dsc)) + '\n')

        HD.append(hd_RV[-1])
        HD.append(hd_Myo[-1])
        HD.append(hd_LV[-1])
        fw.write('hd:' + str(np.mean(HD)) + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_RV' + str(np.mean(Dice_RV)) + '\n')
    fw.write('Dice_Myo' + str(np.mean(Dice_Myo)) + '\n')
    fw.write('Dice_LV' + str(np.mean(Dice_LV)) + '\n')

    fw.write('Mean_hd\n')
    fw.write('hd_RV' + str(np.mean(hd_RV)) + '\n')
    fw.write('hd_Myo' + str(np.mean(hd_Myo)) + '\n')
    fw.write('hd_LV' + str(np.mean(hd_LV)) + '\n')

    fw.write('*' * 20 + '\n')

    dsc = []  # 用于求全部样本的均值
    dsc.append(np.mean(Dice_RV))
    dsc.append(np.mean(Dice_Myo))
    dsc.append(np.mean(Dice_LV))
    fw.write('dsc:' + str(np.mean(dsc)) + '\n')
    print("dsc:", np.mean(dsc))

    HD = []  # 用于求全部样本的均值
    HD.append(np.mean(hd_RV))
    HD.append(np.mean(hd_Myo))
    HD.append(np.mean(hd_LV))
    fw.write('hd:' + str(np.mean(HD)) + '\n')
    print("hd:", np.mean(HD))

    print('done')


if __name__ == '__main__':
    main()
