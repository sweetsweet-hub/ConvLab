import sys
sys.path.insert(0, '..')

import os
import torch

from main import parser
from networks.TDLab_model import TDLabModel


def main():  # 起始行11
    args = parser.parse_args()
    args.logdir = "./runs/" + args.logdir
    model = TDLabModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        norm_name=args.norm_name,
        dropout_path_rate=args.drop_path_rate,
        depths=[3, 3, 5, 3],
        dims=[48, 96, 192, 384]
    )

    # 加载每个模型文件，将状态字典加起来
    state_dict_sum = {}
    for i in [1, 3, 5]:  # *** 改1 ***
        # 加载模型
        model_name = f"model_best{i}.pt"
        model_path = os.path.join(args.pretrained_dir, model_name)
        print("load model_path:", model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        # 将状态字典加起来
        for key, value in state_dict.items():
            if key not in state_dict_sum:
                state_dict_sum[key] = value.clone()
            else:
                state_dict_sum[key] += value
    # 求平均值
    for key in state_dict_sum:
        state_dict_sum[key] /= 3  # *** 改2 ***
    # 将新的状态字典设置给模型
    model.load_state_dict(state_dict_sum)
    # 保存新的模型文件
    ensemble_path = os.path.join(args.logdir, 'ensemble_model_135.pt')  # *** 改3 ***
    torch.save({'state_dict': model.state_dict()}, ensemble_path)
    print("****** successful save ensemble model: ******", ensemble_path)  # 结束行46


if __name__ == "__main__":
    main()
