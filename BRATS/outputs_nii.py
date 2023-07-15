import sys
sys.path.insert(0, '..')

import os
import nibabel as nib
import numpy as np
import torch
#
from monai.inferers import sliding_window_inference

from main import parser
from utils.data_utils import get_loader
from networks.ConvLab_model import ConvLabModel
#

parser.add_argument("--output_directory", type=str, default="./output/output_logname_brats18")


# 输出全部训练集需要改：①data_utils.py第77-78行，②overlap值，③outputs_nii.py第32、36行
#
#
#
#
#
#
#
#


def main():  # 起始行30
    args = parser.parse_args()
    args.test_mode = True  # 对训练集输出用False，对测试集输出用True
    output_directory = args.output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)  # 对训练集输出加[1]，对测试集输出不加
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = ConvLabModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        norm_name=args.norm_name,
        dropout_path_rate=0.0,
        depths=[3, 3, 5, 3],
        dims=[48, 96, 192, 384]
    )

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            val_inputs = batch["image"].cuda()
            original_affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            #
            #
            num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")
            img_name = num[0] + "_" + num[1] + "_" + num[2] + "_" + num[3] + ".nii.gz"
            print("\n------ Inference on case {}".format(img_name), " ------")

            output = torch.sigmoid(sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), args.sw_batch_size, model, overlap=0.8
            ))
            #
            seg = output[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), original_affine), os.path.join(output_directory, img_name))
        print("Finish output.")  # 结束行77


if __name__ == "__main__":
    main()
