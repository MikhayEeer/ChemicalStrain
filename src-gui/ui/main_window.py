'''
主窗口类，负责整体界面布局和交互逻辑
'''
class MainWindow(tk.Tk):
    '''
    主窗口类，负责整体界面布局和交互逻辑
    '''
    def __init__(self):
        super().__init__()
        self.title("Data Visualization Tool")
        self.geometry("800x600")    

        self.create_menu()

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)



