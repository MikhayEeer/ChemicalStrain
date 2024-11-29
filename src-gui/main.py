import tkinter as tk
from tkinter import ttk
import sys
import logging
from pathlib import Path

from ui.main_window import MainWindow
from core.file_handler import FileHandler
from core.data_processor import DataProcessor
from core.plotter import Plotter
from utils.config import Config

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', encoding='utf-8')
        ]
    )

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        tk.messagebox.showerror(
            "错误",
            f"缺少必要的依赖包：{str(e)}\n请安装所需依赖后重试。"
        )
        sys.exit(1)

def create_directories():
    """创建必要的目录结构"""
    directories = [
        'data',
        'output',
        'logs'
    ]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    """主函数"""
    try:
        # 设置日志
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("应用程序启动")

        # 检查依赖
        check_dependencies()
        
        # 创建目录
        create_directories()

        # 加载配置
        config = Config()
        
        # 创建核心组件实例
        file_handler = FileHandler()
        data_processor = DataProcessor()
        plotter = Plotter()

        # 创建并运行主窗口
        root = MainWindow(
            file_handler=file_handler,
            data_processor=data_processor,
            plotter=plotter,
            config=config
        )
        
        # 设置窗口主题样式
        style = ttk.Style()
        style.theme_use('clam')  # 或使用 'alt', 'default', 'classic' 等
        
        # 设置窗口大小和位置
        window_width = 1024
        window_height = 768
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)
        
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # 设置窗口图标（如果有的话）
        # root.iconbitmap('assets/icon.ico')
        
        # 设置窗口最小尺寸
        root.minsize(800, 600)
        
        # 启动事件循环
        logger.info("启动主窗口")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        tk.messagebox.showerror("错误", f"程序运行出错：\n{str(e)}")
        sys.exit(1)
    finally:
        logger.info("应用程序关闭")

if __name__ == "__main__":
    main()