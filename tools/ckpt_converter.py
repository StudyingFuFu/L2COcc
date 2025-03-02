import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ckpt_path = args.ckpt_path
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')

    keys_to_remove = [key for key in ckpt['state_dict'] if 'teacher' in key]
    for key in keys_to_remove:
        del ckpt['state_dict'][key]

    ckpt_name = os.path.basename(ckpt_path)
    save_dir = os.path.dirname(ckpt_path)
    print(ckpt_name)
    output_path = os.path.join(save_dir, 'modified_' + ckpt_name)
    torch.save(ckpt, output_path)
    print(f"Modified checkpoint saved to {output_path}")
    return


if __name__ == '__main__':
    main()
