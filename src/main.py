import os

import dataloader
import utils
import plotters

if __name__ == "__main__":
    print(f"--debug-- 当前工作目录: {os.getcwd()}")
    folder_path = r"samples\4h1%_mini"
    dataloader = dataloader.DataLoader(folder_path)
    dataloader.split_txt_files()
    dataloader.apply_noise_reduction()
    dataloader.phase_correction()

    math_utils = utils.MathUtils(dataloader.get_numbers_txt_files())
    math_utils.get_shear_rate()
    math_utils.fourier_transform()

    plotter = plotters.Plotter(folder_path)

