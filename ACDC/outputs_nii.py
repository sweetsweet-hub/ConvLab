import sys
sys.path.insert(0, '..')

import os
import nibabel as nib
import numpy as np
import torch
import torch.backends.cudnn
from monai.inferers import sliding_window_inference

from main import parser
from utils.data_utils import get_loader
from networks.ConvLab_model import ConvLabModel
from utils.utils import resample_3d

parser.add_argument("--output_directory", type=str, default="./output/output_logname_acdc")

seed = 4492
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def main():  # 起始行30
    args = parser.parse_args()
    args.test_mode = True
    output_directory = args.output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
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
            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            _, h, w, d, _, _, _, _ = batch['image_meta_dict']['dim'].cpu().detach().numpy().tolist()[0]
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            img_name = img_name.replace("img", "infer")
            print("\n------ Inference on case {}".format(img_name), "\ttarget_shape =", target_shape, " ------")

            output = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), args.sw_batch_size, model, overlap=0.8, mode="gaussian"
            )
            print("batch", i, "/", len(val_loader), "转换前output.shape=", output.shape)
            output = torch.softmax(output, 1).cpu().numpy()
            output = np.argmax(output, axis=1).astype(np.uint8)[0]
            output = resample_3d(output, target_shape)
            print("batch", i, "转换后output.shape =", output.shape)
            nib.save(
                nib.Nifti1Image(output.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
            )
        print("Finish output.")  # 结束行77


if __name__ == "__main__":
    main()
