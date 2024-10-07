import re
import os
import glob

class BaseTask:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        _, self.numbers_txt_files = self.generate_txt_files_list(self.folder_path)

    def get_numbers_txt_files(self):
        '''
        获取数字编号的txt文件
        '''
        return self.numbers_txt_files
    
    def generate_txt_files_list(self, folder_path):
        '''
        根据路径生成txt文件的列表
        所有txt列表和 数字txt列表
        '''
        numbers_txt_files = []
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        for file_path in txt_files:
            if os.path.basename(file_path).strip(".txt").isdigit():
                numbers_txt_files.append(file_path)
        return txt_files, numbers_txt_files

    def check_txt_digital_path(self, filename):
        '''
        判断文件是否为 数字.txt file
        '''
        if re.match(r'^\d+\.txt$', filename):
            return True
        return False
    
    def check_file(self, filepath):
        '''
        判断路径是否存在
        '''
        if not os.path.exists(filepath):
            print(f"错误：文件 '{filepath}' 不存在。")
            return False
        return True