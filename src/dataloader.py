import os
import re
import glob
import time
from abstract import BaseTask

class DataLoader(BaseTask):
    def __init__(self, folder_path):
        super().__init__(folder_path)
        print(f"##debug## 数据加载器初始化完成")
        print(f"##debug## 当前数据目录: {self.folder_path}")
        self.start_time = time.time()
    
    def split_txt_files(self):
        print(f"##debug## 开始分割txt文件")

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    datapoint_indices = [i for i, line in enumerate(lines) if line.startswith("Datapoint no.:")]

                    for i, index in enumerate(datapoint_indices):
                        if i == len(datapoint_indices) - 1:
                            data_lines = lines[index + 4:]
                        else:
                            next_index = datapoint_indices[i + 1]
                            data_lines = lines[index + 4:next_index]

                        datapoint_number = lines[index].strip().split(":")[-1].strip()
                        new_filename = f"{datapoint_number.zfill(2)}.txt"
                        new_file_path = os.path.join(self.folder_path, new_filename)

                        #print(f"--debug-- 正在输出到文件: {new_file_path}")
                        with open(new_file_path, 'w') as new_file:
                            for line in data_lines:
                                parts = line.strip().split()
                                new_file.write(f"{parts[0]} {parts[2]} {parts[3]}\n")

        end_time = time.time()
        print(f"##debug## 分割txt文件完成，用时: {end_time - self.start_time}秒")

    def _remove_background_noise(self, input_file):
        try:
            #print(f"--debug-- 正在处理文件: {input_file}")
            with open(input_file, 'r') as f:
                lines = f.readlines()

            max_value_col2 = None
            min_value_col2 = None
            max_value_col3 = None
            min_value_col3 = None

            for line in lines:
                data = line.strip().split()
                if len(data) >= 3:
                    value_col2 = float(data[1])
                    value_col3 = float(data[2])
                    if max_value_col2 is None or value_col2 > max_value_col2:
                        max_value_col2 = value_col2
                    if min_value_col2 is None or value_col2 < min_value_col2:
                        min_value_col2 = value_col2
                    if max_value_col3 is None or value_col3 > max_value_col3:
                        max_value_col3 = value_col3
                    if min_value_col3 is None or value_col3 < min_value_col3:
                        min_value_col3 = value_col3

                a_col2 = (max_value_col2 + min_value_col2) / 2
                a_col3 = (max_value_col3 + min_value_col3) / 2

            processed_lines = []
            for line in lines:
                data = line.strip().split()
                if len(data) >= 3:
                    value_col1 = float(data[0])
                    value_col2 = float(data[1])
                    value_col3 = float(data[2])
                    new_value_col2 = value_col2 - a_col2
                    new_value_col3 = value_col3 - a_col3
                    processed_lines.append(f"{value_col1} {new_value_col2} {new_value_col3}\n")

            with open(input_file, 'w') as f:
                f.writelines(processed_lines)
        
        except Exception as e:
            if "Permission denied" in str(e):
                pass
            else:
                raise e
    
    def apply_noise_reduction(self):
        txt_files = glob.glob(os.path.join(self.folder_path, '*.txt'))
        print(f"##debug## 开始背景噪声消除")
        for file_path in txt_files:
            if os.path.basename(file_path).strip(".txt").isdigit():
                self._remove_background_noise(file_path)

    def phase_correction(self):
        print(f"##debug## 开始相位矫正")
        for filename in os.listdir(self.folder_path):
            if self.check_txt_digital_path(filename):
                input_file_path = os.path.join(self.folder_path, filename)

                data = []
                with open(input_file_path, 'r') as file:
                    for line in file:
                        values = line.strip().split()
                        data.append([float(values[0]), float(values[1]), float(values[2])])

                max_strain_index = max(range(len(data)), key=lambda i: data[i][2])
                max_strain_value = data[max_strain_index][2]

                shift_amount = 128 - max_strain_index

                shifted_data = []
                for i in range(512):
                    source_index = (i - shift_amount) % 512
                    shifted_data.append([i, data[source_index][1], data[source_index][2]])
                
                #print(f"--debug-- 正在写入矫正后的数据 {input_file_path}")
                with open(input_file_path, 'w') as output_file:
                    for row in shifted_data:
                        output_file.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                
                with open(input_file_path, 'r') as file:
                    shifted_data_check = []
                    for line in file:
                        values = line.strip().split()
                        shifted_data_check.append([int(values[0]), float(values[1]), float(values[2])])
                    
                    if len(shifted_data_check) != 512:
                        raise ValueError(f"##error## 矫正后的数据长度不等于512: {len(shifted_data_check)}")
                    
                    if any(abs(shifted_data_check[i][2] - shifted_data[i][2]) > 0.01 for i in range(512)):
                        raise ValueError("##error## 矫正后的数据与预期不符")
                    
                    
