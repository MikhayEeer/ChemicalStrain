import os
import glob
import pandas as pd
import numpy as np
import re
from scipy.optimize import curve_fit
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


#输入文件夹路径
folder_path =r"C:\Users\mikhayeeer\Documents\WeChat Files\wxid_cwx3dxsz49ut21\FileStorage\File\2024-09\4h 1%"

plt.rcParams['font.family'] = 'Times New Roman'  # Matplotlib 默认字体，支持大部分 Unicode 字符
def split_txt_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # 获取 "Datapoint no.: " 的索引
                datapoint_indices = [i for i, line in enumerate(lines) if line.startswith("Datapoint no.:")]

                for i, index in enumerate(datapoint_indices):
                    if i == len(datapoint_indices) - 1:
                        # 最后一个 "Datapoint no.: " 到文件末尾的数据
                        data_lines = lines[index + 4:]  # 忽略前三行和 "Datapoint no.: " 行
                    else:
                        # 当前 "Datapoint no.: " 到下一个 "Datapoint no.: " 之前的数据
                        next_index = datapoint_indices[i + 1]
                        data_lines = lines[index + 4:next_index]  # 忽略前三行和 "Datapoint no.: " 行

                    # 创建新的文件名
                    datapoint_number = lines[index].strip().split(":")[-1].strip()
                    new_filename = f"{datapoint_number.zfill(2)}.txt"
                    new_file_path = os.path.join(folder_path, new_filename)

# 将数据写入新文件，只保留第1、第3和第4列的数据
                    with open(new_file_path, 'w') as new_file:
                        for line in data_lines:
                            parts = line.strip().split()
                            new_line = ' '.join([parts[0], parts[2], parts[3]]) + '\n'
                            new_file.write(new_line)

split_txt_files(folder_path)



# 定义背景消除函数
def process_txt_file(input_file):
    try:
        # 打开输入文件
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # 初始化第二列和第三列的最大值和最小值
        max_value_col2 = None
        min_value_col2 = None
        max_value_col3 = None
        min_value_col3 = None

        # 找到第三列和第四列数据的最大值和最小值
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

                # 计算a
            a_col2 = (max_value_col2 + min_value_col2) / 2
            a_col3 = (max_value_col3 + min_value_col3) / 2

        # 存储处理后的数据
        processed_lines = []
        for line in lines:
            data = line.strip().split()
            if len(data) >= 3:
                value_col1 = float(data[0])
                value_col2 = float(data[1])
                value_col3 = float(data[2])
                new_value_col2 = value_col2 - a_col2
                new_value_col3 = value_col3 - a_col3
                processed_lines.append(f'{value_col1} {new_value_col2} {new_value_col3}\n')

        # 覆盖原始文件中的内容
        with open(input_file, 'w') as f:
            f.writelines(processed_lines)

    except Exception as e:
        # 捕获异常，忽略 "Permission denied" 错误
        if "Permission denied" in str(e):
            pass
        else:
            print(f"An error occurred while processing {input_file}: {str(e)}")


txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

# 遍历每个txt文件并处理
for file_path in txt_files:
    # 检查文件名是否为纯数字
    file_name = os.path.basename(file_path)
    if file_name[:-4].isdigit():  # 去掉后缀".txt"再检查是否为纯数字
        process_txt_file(file_path)



#相位校正

# 遍历输入文件夹中的所有txt文件
for filename in os.listdir(folder_path):
    if re.match(r'^\d+\.txt$', filename):
        input_file_path = os.path.join(folder_path, filename)

        data = []
        with open(input_file_path, 'r') as file:
            for line in file:
                values = line.strip().split()
                data.append([float(values[0]), float(values[1]), float(values[2])])

        # 找到应变的最大值及其索引
        max_strain_index = max(range(len(data)), key=lambda i: data[i][2])
        max_strain_value = data[max_strain_index][2]  # 保存最大应变值以便检查

        shift_amount = 128 - max_strain_index  # 确保最大应变值校准到第128个数处

        shifted_data = []
        for i in range(512):
            source_index = (i - shift_amount) % 512
            shifted_data.append([i, data[source_index][1], data[source_index][2]])

        # 写入矫正后的数据
        with open(input_file_path, 'w') as output_file:
            for row in shifted_data:
                output_file.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")

        # 调试：检查结果是否正确
        with open(input_file_path, 'r') as file:
            shifted_data_check = []
            for line in file:
                values = line.strip().split()
                shifted_data_check.append([int(values[0]), float(values[1]), float(values[2])])

#得到剪切速率值

# 定义sin函数模型
def sin_func(x, A, omega, phi, C):
    return A * np.sin(omega * x + phi) + C

# 定义sin函数的导数
def derivative(x, A, omega, phi, C):
    return omega * A  * np.cos(omega * x + phi)


# 获取文件夹中所有名字为纯数字的 txt 文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and f[:-4].isdigit()]

for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)

    # 读取txt文件并提取数据
    data = np.loadtxt(file_path)
    x = data[:, 0]/512
    y = data[:, 2]

    # 估计A0和omega0的初始值
    A0 = (np.max(y) - np.min(y)) / 2
    omega0 = 2 * np.pi / np.max(x)

    # 使用curve_fit进行拟合，不提供初始参数
    params, covariance = curve_fit(sin_func, x, y, p0=[A0, omega0, 0, 0])
    A, omega, phi, C = params

    new_y = sin_func(x, A, omega, phi, C)
    data[:, 2] = new_y

    # 计算导数值
    derivatives = derivative(x, A, omega, phi, C)

    # 将导数值作为第四列添加到原始数据
    data_with_derivatives = np.column_stack((data, derivatives))

    # 将包含导数值的数据写回原始txt文件
    np.savetxt(file_path, data_with_derivatives, delimiter='\t')

#傅里叶变换
output_folder = os.path.join(folder_path, "fourier_coefficients")

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取源文件夹中的所有txt文件名为纯数字的文件
txt_files = [file for file in glob.glob(os.path.join(folder_path, '*.txt')) if
                 re.match(r'^\d+\.txt$', os.path.basename(file))]

# 遍历每个txt文件
for txt_file in txt_files:
    # 读取txt文件内容并提取第三列数据
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    data = []
    strain=[]
    for line in lines:
        columns = line.split()
        if len(columns) >= 3:
            data.append(float(columns[1]))
            strain.append(float(columns[2]))

    # 进行傅里叶变换
    data_array = np.array(data, dtype=float)
    fourier_coefficients = np.fft.fft(data_array) * 2 / len(data_array)
    # 计算前十项的奇数项傅里叶系数实部和虚部平方和并开根号

    result = []
    for n in range(1, 15):  # 计算前十项
        C_n = fourier_coefficients[n]
        I_n = np.sqrt(np.abs(C_n.real) ** 2 + np.abs(C_n.imag) ** 2)
        real_part = np.abs(C_n.real)
        imag_part = np.abs(C_n.imag)
        result.append((I_n, C_n.real, -C_n.imag))  # 添加I_n和C_n到结果列表中

    # 构建输出文件的完整路径，保持原始文件名不变
    output_filepath = os.path.join(output_folder, os.path.basename(txt_file))

    # 将结果写入新txt文件
    with open(output_filepath, 'w') as output_file:
        for i, (I_n, real_part, imag_part) in enumerate(result):
            output_file.write(f" {I_n}, {real_part},{imag_part}\n")



# 输入选择，如果要绘制李萨如环，则输入1
choice = input("绘制李萨如环请输入1，应力应变归一化输入2，李萨如环叠加请输入3，ST值请输入4，同一应变不同In/I1比较请输入5，\n不同应变同一In/I1比较请输入6，SPP方法请输入7，应力分叉请输入8，耗散率请输入9，笼子模量请输入10，Spp绘图程序请输入11（请确保运行“7”程序完成数据计算）：")

if choice == "1":
    # 输入数字来构建文件名
    file_number = input("请输入文件编号：")
    file_name = os.path.join(folder_path, f"{file_number}.txt")

    # 从txt文件加载数据
    data = np.loadtxt(file_name)
    # 提取第二列、第三列和第四列
    stress = data[:, 1]  # 第二列
    strain = data[:, 2]  # 第三列
    strain_rate = data[:, 3]  # 第四列

    colors ="haline_r"


    # 创建3D散点图
    fig_3d = go.Figure(data=[go.Scatter3d(x=strain, y=strain_rate, z=stress, mode='markers',
                                          marker=dict(size=6, color=stress, colorscale=colors, opacity=0.8))])

    # 设置图形布局
    fig_3d.update_layout(scene=dict(xaxis_title='Strain',
                                     yaxis_title='Strain Rate',
                                     zaxis_title='Stress',
                                     xaxis=dict(showgrid=False),
                                     yaxis=dict(showgrid=False),
                                     zaxis=dict(showgrid=False),
                                    aspectmode='cube'),
                          title=f'Stress-Strain Rate-Strain Relationship (File Number: {file_number})')

    # 创建 x、y、z 轴投影图
    fig_projections = make_subplots(rows=1, cols=3,
                                    subplot_titles=[f' Elastic Lissajous Curve (File Number: {file_number})',
                                                    f' Viscous Lissajous Curve (File Number: {file_number})',
                                                    f'Z-axis Projection (File Number: {file_number})'])

    # 添加散点图到投影图
    fig_projections.add_trace(go.Scatter(x=strain, y=stress, mode='markers',
                                         marker=dict(size=12, color=stress, colorscale=colors, opacity=0.8),
                                         line=dict(width=5)),
                               row=1, col=1)
    fig_projections.add_trace(go.Scatter(x=strain_rate, y=stress, mode='markers',
                                         marker=dict(size=12, color=stress, colorscale=colors, opacity=0.8),
                                         line=dict(width=5)),
                               row=1, col=2)
    fig_projections.add_trace(go.Scatter(x=strain, y=strain_rate, mode='markers',
                                         marker=dict(size=12, color=stress, colorscale=colors, opacity=0.8),
                                         line=dict(width=5)),
                               row=1, col=3)

    # 更新投影图的布局，去除坐标轴
    fig_projections.update_layout(title='Stress-Strain Rate-Strain Projections',
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   xaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   xaxis3=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis3=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)',
                                  autosize=False,  # 禁用自动调整大小
                                  width=1500,  # 设置图表宽度
                                  height=600)  # 设置图表高度


    # 在 PyCharm 中显示图形
    pio.show(fig_3d)
    pio.show(fig_projections)

elif choice == "2":
    # 归一化

    new_folder_name = "归一化结果"
    # 构建新文件夹的完整路径
    normalization_folder_path = os.path.join(folder_path, new_folder_name)
    # 使用 os.mkdir() 函数创建新文件夹
    os.makedirs(normalization_folder_path, exist_ok=True)

    # 遍历输入文件夹中的所有 txt 文件名为纯数字的文件
    for filename in os.listdir(folder_path):
        if re.match(r'^\d+\.txt$', filename):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(normalization_folder_path, filename)

            # 读取 txt 文件数据
            data = np.loadtxt(input_file_path)

            # 找到每列的最大值
            max_stress = np.max(data[:, 1])
            max_strain = np.max(data[:, 2])
            max_strain_rate = np.max(data[:, 3])

            # 将每列数据除以对应列的最大值，实现归一化
            data[:, 1] /= max_stress
            data[:, 2] /= max_strain
            data[:, 3] /= max_strain_rate

            np.savetxt(output_file_path, data, fmt='%f', delimiter='\t')  # 以制表符分隔保存数据

elif choice == "3":

    def plot_graphs(file_paths):
        # 初始化两个图表
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        for file_path in file_paths:
            # 读取txt文件
            data = np.loadtxt(file_path, skiprows=1)

            # 提取数据列
            strain = data[:, 2]
            stress = data[:, 1]
            strain_rate = data[:, 3]

            # 绘制第一个图：stress vs. strain
            ax1.plot(strain, stress, label=f'{os.path.basename(file_path)}')

            # 绘制第二个图：stress vs. strain_rate
            ax2.plot(strain_rate, stress, label=f'{os.path.basename(file_path)}')

        # 设置第一个图表的标题和标签
        ax1.set_xlabel('Strain')
        ax1.set_ylabel('Stress')
        ax1.set_title('Stress vs. Strain')
        # 使用对数坐标

        ax1.legend()

        # 设置第二个图表的标题和标签
        ax2.set_xlabel('Strain Rate')
        ax2.set_ylabel('Stress')
        ax2.set_title('Stress vs. Strain Rate')
        # 使用对数坐标

        ax2.legend()

        # 显示图表
        plt.show()


    # 获取文件夹中的所有txt文件
    txt_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                 re.match(r'^\d+\.txt$', filename)]

    # 输入要选择的文件编号（逗号分隔）
    selected_indices_str = input("输入要选择的文件编号（空格分隔）：")

    # 解析用户输入的编号
    selected_indices = [int(idx) - 1 for idx in selected_indices_str.split(' ')]

    # 根据选择的编号获取对应的文件路径
    selected_files = [txt_files[idx] for idx in selected_indices]

    # 绘制图表
    plot_graphs(selected_files)


elif choice == "4":

    # 指定文件夹路径
    excel_folder_path = folder_path
    # 初始化stress列表
    strain_value = []

    # 获取文件夹中的所有文件
    files1 = os.listdir(excel_folder_path)

    # 筛选出Excel文件
    excel_files = [file for file in files1 if file.endswith('.xlsx') or file.endswith('.xls')]

    # 遍历每个Excel文件
    for excel_file in excel_files:
        # 构建文件路径
        file_path = os.path.join(excel_folder_path, excel_file)

        # 读取Excel文件中的数据，假设第一个工作表中包含数据
        df = pd.read_excel(file_path)

        # 获取第三列数据
        strain_column = df["ɣ in -"]
        # 将第三列数据添加到stress_value列表
        strain_value.extend(strain_column)

    F_folder_path = os.path.join(folder_path, "fourier_coefficients")
    # 初始化总的stress和strain列表
    S_value = []
    T_value = []
    G_l_value=[]
    G_m_value=[]
    eta_l_value=[]
    eta_m_value = []
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(F_folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(F_folder_path, filename)

            # 初始化当前文件的stress和strain列表
            real = []
            imag = []

            # 打开当前文件
            with open(file_path, 'r') as file:
                # 逐行读取文件内容
                for line in file:
                    # 分割每行数据
                    data = line.strip().split(',')

                    # 如果数据行包含至少三列
                    if len(data) >= 3:
                        # 第二列数据添加到stress列表
                        real.append(float(data[1]))

                        # 第三列数据添加到strain列表
                        imag.append(float(data[2]))

                # 计算当前文件的 ∑(n+1)*stress[n] 和 ∑(n+1)*strain[n]
                ηl = 0
                ηm = 0
                Gm = 0
                Gl = 0
                for i in range(0, len(real), 2):
                    ηl += real[i]
                    ηm += (i + 1) * real[i] * ((-1) ** (i / 2))
                    Gm += (i + 1) * imag[i]
                    Gl += imag[i] * ((-1) ** (i / 2))

            # 避免除以零错误
            if Gl != 0:
                # 计算S
                S = (Gl - Gm) / Gl
            else:
                S = 0

            if ηl != 0:
                # 计算T
                T = (ηl - ηm) / ηl
            else:
                T = 0

            # 将当前文件的结果添加到总的stress和strain列表

            S_value.append(S)
            T_value.append(T)
            G_l_value.append(Gl)
            G_m_value.append(Gm)
            eta_l_value.append(ηl)
            eta_m_value.append(ηm)

    # 将结果保存到新的txt文件

    output_file_path = os.path.join(folder_path, "ST.txt")
    with open(output_file_path, 'w') as output_file:
        for S, T, G_l, G_m, eta_l, eta_m in zip(S_value, T_value,G_l_value,G_m_value,eta_l_value,eta_m_value):
            output_file.write(f"{S}\t{T}\t{G_l}\t{G_m}\t{eta_l}\t{eta_m}\n")

    print(f"结果已保存到 {output_file_path}")





    # 读取ST.txt文件
    input_file_path = os.path.join(folder_path, "ST.txt")
    Gl, Gm, el, em = [], [], [], []

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            values = line.strip().split('\t')
            Gl.append(float(values[2]))
            Gm.append(float(values[3]))
            el.append(float(values[4]))
            em.append(float(values[5]))

    # 计算mod_Gl和其他修正值
    mod_Gl = [g / s for g, s in zip(Gl, strain_value)]
    mod_Gm = [g / s for g, s in zip(Gm, strain_value)]
    mod_el = [e / (s*omega) for e, s in zip(el, strain_value)]
    mod_em = [e / (s*omega) for e, s in zip(em, strain_value)]

    # 保存修正后的值到新的txt文件
    output_mod_file_path = os.path.join(folder_path, "mod_ST.txt")
    with open(output_mod_file_path, 'w') as output_mod_file:
        for mg, mm, mel, mem in zip(mod_Gl, mod_Gm, mod_el, mod_em):
            output_mod_file.write(f"{mg}\t{mm}\t{mel}\t{mem}\n")

    print(f"Gl、Gm、etal、etam已保存到 {output_mod_file_path}")

    # 绘制第一张图（stress vs S）
    plt.figure(figsize=(10, 5))
    plt.scatter(strain_value, S_value, color='blue', marker='o', label='S')
    plt.title('Strain vs S')
    plt.xscale('log')
    plt.xlabel('Stress')
    plt.ylabel('S')
    plt.grid(False)  # 去掉背景网格

    # 绘制第二张图（stress vs T）
    plt.figure(figsize=(10, 5))
    plt.scatter(strain_value, T_value, color='red', marker='o', label='T')
    plt.title('Strain vs T')
    plt.xscale('log')
    plt.xlabel('Stress')
    plt.ylabel('T')
    plt.grid(False)  # 去掉背景网格
    plt.show()

elif choice =="5":


    F_folder_path = os.path.join(folder_path, "fourier_coefficients")
    # 列出文件夹中的所有txt文件
    txt_files = [f for f in os.listdir(F_folder_path) if f.endswith('.txt')]



    # 自定义排序函数，按文件编号排序
    def custom_sort(file_name):
        try:
            return int(file_name.split('.')[0])
        except ValueError:
            return float('inf')


    # 按文件编号进行排序
    txt_files.sort(key=custom_sort)

    # 读取Excel文件中第三列的数据作为stress
    # 假设你已经定义了以下变量：
    # new_folder_path

    # 存储所有找到的 Excel 文件路径
    excel_file_paths = []

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否为 Excel 文件（可以根据实际情况调整）
            if file.endswith(".xlsx") or file.endswith(".xls"):
                # 构建完整的文件路径
                excel_file_path = os.path.join(root, file)
                excel_file_paths.append(excel_file_path)

    df = pd.read_excel(excel_file_path)
    strain_values = df['ɣ in -'].tolist()

    # 打印文件编号供用户选择
    for i, file_name in enumerate(txt_files):
        print(f"{i + 1}. {file_name}")

    # 输入要选择的文件编号（逗号分隔）
    selected_indices_str = input("输入要选择的文件编号（空格分隔）：")

    # 将输入的逗号分隔的字符串转换为整数列表
    selected_indices = [int(idx) - 1 for idx in selected_indices_str.split(' ') if idx.strip().isdigit()]

    # 创建 3D 折线图
    fig = go.Figure()
    # 存储每个文件编号的 x、y、z 值
    file_data = []

    # 遍历选择的txt文件并添加到图中
    for i, idx in enumerate(selected_indices):
        if 0 <= idx < len(txt_files):
            file_name = txt_files[idx]
            file_path = os.path.join(F_folder_path, file_name)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 提取第一列数据
            y_values = []

            for line in lines:
                # 拆分逗号分隔的数字，并转换为浮点数
                values = [float(value) for value in line.strip().split(',')]
                # 选择第一个数字作为y值
                y_values.append(values[0])

            # x轴为数据行数
            x_values = list(range(1, len(y_values) + 1))

            # 调整y_values，使每个值都除以y_values中的最大值
            max_y_value = max(y_values)
            y_values = [value / max_y_value for value in y_values]

            # 修改y_values为对应的stress值
            strain_value = strain_values[idx]
            # 将 x、y、z 值添加到文件数据列表中
            file_data.append((file_name, x_values, [strain_value] * len(x_values), y_values))
            # 打印每个文件编号的 x、y、z 值
            for file_name, x_values, y_values, z_values in file_data:
                print(f"文件名: {file_name}")
                print("Harmonic Strain In/I1:")
                for x, y, z in zip(x_values, y_values, z_values):
                    print(f"{x}\t{y}\t{z}")
                print()

            y_values_scaled = [y * 100 for y in y_values]



            # 添加折线图到图中
            fig.add_trace(go.Scatter3d(
                x=x_values,
                y=y_values_scaled,
                z=z_values,
                mode='lines',  # 使用折线模式
                line=dict(width=6),
                name=file_name
            ))
    # 设置图形布局
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Harmonic', tickmode='auto'),
            yaxis=dict(
                title='Strain (%)',
                type="log",
                tickmode='linear',
                tickvals=[0.1, 1, 10, 100, 200, 500, 1000, 2000],  # 手动指定刻度值
                ticktext=["0.1","1",'10', '100',"200", '500', '1000',"2000"],  # 对应的显示文本
                dtick=1,
                tickformat=''  # y 轴保留五位有效数字的百分比显示
            ),
            zaxis=dict(
                title='In/I1',
                type="log",
                tickmode='array',  # 使用自定义刻度
                tickvals=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],  # 指定实际的值
                ticktext=['10\u207B\u2075', '10\u207B\u2074', '10\u207B\u00B3', '10\u207B\u00B2', '10\u207B\u00B9',
                          '10\u2070'],  # 上标形式的刻度标签
                dtick=1
            )
        ),
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=1, y=1, z=1),  # 设置等比例
        title='3D 折线图',
    )

    # 显示图形
    fig.show()






elif choice =="6":


    def calculate_ratio(file_path, n):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            first_line_data = float(lines[0].strip().split(",")[0])  # 读取第一行数据的第一列值
            nth_line_data = float(lines[n - 1].strip().split(",")[0])  # 读取第n行数据的第一列值
            ratio = nth_line_data / first_line_data
            return ratio


    F_folder_path = os.path.join(folder_path, "fourier_coefficients")

    # 存储所有找到的 Excel 文件路径
    excel_file_paths = []

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否为 Excel 文件（可以根据实际情况调整）
            if file.endswith(".xlsx") or file.endswith(".xls"):
                # 构建完整的文件路径
                excel_file_path = os.path.join(root, file)
                excel_file_paths.append(excel_file_path)

    df = pd.read_excel(excel_file_path)
    strain_values = df['ɣ in -'].tolist()


    # 假设 strain_values 已经有了相应的数据
    df["strain"] = strain_values

    try:
        n = int(input("请输入n的值："))  # 输入n的值
    except ValueError:
        print("错误：n必须是一个整数")
        exit()

    ratios = []

    # 遍历文件夹中的所有 txt 文件
    for filename in os.listdir(F_folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(F_folder_path, filename)
            try:
                ratio = calculate_ratio(file_path, n)  # 假设 calculate_ratio 是你定义的函数
                ratios.append(ratio)
            except IndexError:
                print(f"文件 '{filename}'：错误：n超出文件行数范围")
            except Exception as e:
                print(f"文件 '{filename}'：错误：{e}")

    # 检查 strain_values 和 ratios 是否长度相同
    if len(strain_values) != len(ratios):
        print("错误：应变数据和比率数据的长度不匹配")
        exit()

    # 将 strain 和 ratios 组合成 DataFrame 并打印
    results_df = pd.DataFrame({
        'Strain': strain_values,
        f'I{n}/I1 Ratio': ratios
    })

    # 打印 strain 和 ratio 数据
    print(results_df.to_string(index=False))

    # 绘制图表
    plt.plot(strain_values, ratios, marker='o', linestyle='-')
    plt.xlabel('Strain')
    plt.ylabel(f'I{n}/I1')

    # 将纵轴设置为对数尺度
    plt.xscale('log')
    plt.yscale('log')

    plt.title(f'Strain vs. I{n}/I1')
    plt.grid(True)
    plt.show()


elif choice =="7":
   # 频率
    om = 2 * np.pi

    # 获取原始文件夹名
    folder_name = os.path.basename(folder_path)

    # 输入数字来构建文件名
    file_number = input("请输入文件编号：")
    file_name = os.path.join(folder_path, "fourier_coefficients", f"{file_number}.txt")

    # 打开傅里叶系数txt文件
    with open(file_name, 'r') as file:
        lines = file.readlines()

    N = int(input("请输入N："))

    print(f"选择的文件编号为:{file_number}，选择的傅里叶级数为:{N}")
    real_values = []
    imag_values = []

    for i in range(0, N):
        line = lines[i].strip().split(",")
        real_values.append(float(line[1]))  # 第二列数据
        imag_values.append(float(line[2]))  # 第三列数据


    # s(t)

    def r(t, real_values, imag_values, om, N):
        result_1 = 0
        for n in range(0, N + 1,2):  # N只取奇数
            term = real_values[n] * np.cos((n + 1) * om * t) + imag_values[n] * np.sin((n + 1) * om * t)
            result_1 += term
        return result_1


    t = np.linspace(0, 1, 512)  # 创建时间点
    stress_values = [r(time, real_values, imag_values, om, N) for time in t]  # 计算r(t)的值


    # 读取第一个 txt 文件
    input_file_path = os.path.join(folder_path, f"{file_number}.txt")

    with open(input_file_path, 'r') as input_file:
        data = input_file.readlines()

    # 提取第二列数据
    stress_values_old = [line.split()[1] for line in data]
    strain_values = [line.split()[2] for line in data]
    strain_rate_values = [line.split()[3] for line in data]



    # 创建 "rebuild" 文件夹
    rebuild_folder_path = os.path.join(folder_path, "rebuild")
    os.makedirs(rebuild_folder_path, exist_ok=True)

    # 写入新的 txt 文件，包括第二列数据和 r(t) 值
    output_file_path = os.path.join(rebuild_folder_path, f"{file_number}_rebuild.txt")
    with open(output_file_path, 'w') as output_file:
        for i in range(len(strain_values)):
            output_file.write(f"{t[i]}\t{stress_values[i]}\t{strain_values[i]}\t{strain_rate_values[i]}\n")

    # 读取txt文件

    data_new = np.loadtxt(output_file_path)

    # 分别提取时间、strain、strain_rate、stress数据列

    data_read_num =256
    t =data_new[:data_read_num,0]

    # 读取 stress 列的前一半行
    stress = data_new[:data_read_num, 1]

    # 读取 strain 列的前一半行
    strain = data_new[:data_read_num, 2]

    # 读取 strain_rate 列的前一半行
    strain_rate = data_new[:data_read_num, 3] / om
    # 构建向量P
    P = np.column_stack((strain, strain_rate, stress))

    # 计算P对时间的导数，即P'(t)
    P_prime = np.gradient(P, t, axis=0)



    # 计算P'(t)的模
    norm_P_prime = np.linalg.norm(P_prime, axis=1)
    norm_P_prime[norm_P_prime == 0] = np.finfo(float).eps  # 避免零除错误

    # 将P'(t)的每个分量除以其模的绝对值，得到单位向量T
    T = P_prime / norm_P_prime[:, np.newaxis]

    # 计算T对时间的导数，即T'(t)
    T_prime = np.gradient(T, t, axis=0)

    # 计算T'(t)的模
    norm_T_prime = np.linalg.norm(T_prime, axis=1)
    norm_T_prime[norm_T_prime == 0] = np.finfo(float).eps  # 避免零除错误

    # 将T'(t)除以其模，得到单位向量N
    N = T_prime / norm_T_prime[:, np.newaxis]

    # 计算T和N的叉乘，得到向量B
    B = np.cross(T, N)

    # 计算B在γ和σ方向上的投影
    B_strain = np.dot(B, np.array([1, 0, 0]))
    B_strain_rate = np.dot(B, np.array([0, 1, 0]))
    B_stress = np.dot(B, np.array([0, 0, 1]))

    # 计算G'(t) = -B_gamma(t) / B_sigma(t)
    G1 = -B_strain / B_stress

    # 计算G''(t) = -B_gamma_dot / (omega * B_sigma)
    G2 = -B_strain_rate / B_stress

    # 计算G'(t)对时间的导数，即G'(t)/dt
    G1_prime = np.gradient(G1,t, axis=0)  # 使用numpy.gradient计算导数
    G2_prime = np.gradient(G2,t, axis=0)

    # 创建 "SPP Datd" 文件夹
    rebuild_folder_path_2 = os.path.join(folder_path, "SPP Data")
    os.makedirs(rebuild_folder_path_2, exist_ok=True)

    # 写入新的 txt 文件
    output_file_path_2 = os.path.join(rebuild_folder_path_2, f"{file_number}_spp data.txt")

    with open(output_file_path_2, 'w') as output_file:
        for i in range(len(G1)):
            output_file.write(f"{G1[i]}\t{G2[i]}\t{G1_prime[i]}\t{G2_prime[i]}\n")


    # 创建颜色渐变
    colors1 = plt.cm.viridis(stress / max(stress))

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # 绘制第一个散点图
    scatter1 = axs[0].scatter(G1, G2, c=colors1, s=50, alpha=0.8, edgecolors='w', linewidth=0.5)
    axs[0].set_title('Cole–Cole plots of Gₜ\'(t)and Gₜ″(t)')
    axs[0].set_xlabel('Gₜ\'(t)')
    axs[0].set_ylabel('Gₜ″(t)')
    axs[0].grid(True)

    # 添加颜色条
    cbar1 = fig.colorbar(scatter1, ax=axs[0])
    cbar1.set_label('Data Value')

    # 绘制第二个散点图
    scatter2 = axs[1].scatter(G1_prime, G2_prime, c=colors1, s=50, alpha=0.8, edgecolors='w', linewidth=0.5)
    axs[1].set_title('time-derivatives of transient moduli')
    axs[1].set_xlabel('dGₜ\'(t)/dt')
    axs[1].set_ylabel('dGₜ″(t)/dt')
    axs[1].grid(True)

    # 添加颜色条
    cbar2 = fig.colorbar(scatter2, ax=axs[1])
    cbar2.set_label('Data Value')

    # 调整子图布局
    plt.tight_layout()

    # 显示图形
    plt.show()


elif choice == "8":
    # 做应力分叉图

    # 初始化存储最大值和对应数据的列表
    max_a_values = []
    max_b_values = []
    max_c_values = []
    d_values = []
    e_values = []
    f_values = []
    g_values = []


    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(folder_path):
        if re.match(r'^\d+\.txt$', filename):
            file_path = os.path.join(folder_path, filename)

            # 读取txt文件中的数据，假设数据以空格分隔
            data = np.loadtxt(file_path, delimiter='\t')

            # 提取每一列数据
            column2 = data[:, 1]  # 第二列数据
            column3 = data[:, 2]  # 第三列数据
            column4 = data[:, 3]  # 第四列数据

            # 计算每列数据的最大值
            max_a = np.max(column2)
            max_b = np.max(column3)
            max_c = np.max(column4)

            # 将最大值添加到列表中
            max_a_values.append(max_a)
            max_b_values.append(max_b)
            max_c_values.append(max_c)

            # 找到最大值对应的数据
            max_a_index = np.argmax(column2)
            max_b_index = np.argmax(column3)
            max_c_index = np.argmax(column4)

            d_values.append(column3[max_a_index])
            e_values.append(column2[max_b_index])
            f_values.append(column2[max_c_index])
            g_values.append(column4[max_a_index])

    # 创建一个新的txt文件来保存所有值
    output_file_path = routput_file_path = os.path.join(folder_path, "应力分叉.txt")
    with open(output_file_path, 'w') as output_file:
        for a, b, c, d, e, f, g in zip(max_a_values, max_b_values, max_c_values, d_values, e_values, f_values,
                                       g_values):
            output_file.write(f"{a} {b} {c} {d} {e} {f} {g}\n")

#第一列是σmax，第二列是γ0，第三列是γ ̇ 0，第四列是γσmax，第五列是σγ0，第六列是γ ̇ σmax，第七列是σγ ̇ 0

    # 做应力分叉图
    # 读取txt文件中的数据，假设数据以空格分隔
    data = np.loadtxt(output_file_path, delimiter=' ')

    # 提取数据列
    x1 = data[:, 3]  # 第四列数据作为x1变量
    y1 = data[:, 0]  # 第一列数据作为y1变量

    x2 = data[:, 1]  # 第二列数据作为x2变量
    y2 = data[:, 0]  # 第一列数据作为y2变量

    x3 = data[:, 1]  # 第二列数据作为x3变量
    y3 = data[:, 4]  # 第五列数据作为y3变量

    # 创建第一张图，包含三条曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # 创建第一个子图

    # 绘制第一条散点图，并设置散点的样式和颜色
    plt.scatter(x1, y1, label='σ_max~γ ̅_max', color='blue', marker='o', s=40)
    # 绘制第二条散点图，设置不同的散点样式和颜色
    plt.scatter(x2, y2, label='σ ̅_max~γ_max', color='red', marker='o', s=40)
    # 绘制第三条散点图，设置不同的散点样式和颜色
    plt.scatter(x3, y3, label='σ_max~γ_max', color='green', marker='o', s=40)
    plt.xlabel('Strain')
    plt.ylabel('Stress(Pa)')
    plt.title('Three Curves in the First Plot')
    plt.legend()

    # 设置x轴和y轴为对数坐标
    plt.xscale('log')
    plt.yscale('log')
    x1_min =max(np.min(x1),1e-6)
    x1_max=np.max(x1)
    y1_min = max(np.min(y1),1e-6)
    y1_max = np.max(y1)

    # 设置x轴的范围，取数据范围的1/10和10倍
    plt.xlim(x1_min /10, x1_max * 10)

    # 设置y轴的范围，取数据范围的1/10和10倍
    plt.ylim(y1_min /10, y1_max * 10)

    # 提取第二组数据
    x4 = data[:, 6]  # 第七列数据作为x4变量
    y4 = data[:, 0]  # 第一列数据作为y4变量

    x5 = data[:, 2]  # 第三列数据作为x5变量
    y5 = data[:, 0]  # 第一列数据作为y5变量

    x6 = data[:, 2]  # 第三列数据作为x6变量
    y6 = data[:, 5]  # 第六列数据作为y6变量

    # 创建第二张图，包含三条曲线
    plt.subplot(1, 2, 2)  # 创建第二个子图
    plt.scatter(x4, y4, label='σ_max~γ ̇  ̅_max', color='blue', marker='o', s=40)
    # 绘制第二条散点图，设置不同的散点样式和颜色
    plt.scatter(x5, y5, label='σ̇  ̅_max~γ ̇ _max', color='red', marker='o', s=40)
    # 绘制第三条散点图，设置不同的散点样式和颜色
    plt.scatter(x6, y6, label='σ̇_max~γ ̇ _max', color='green', marker='o', s=40)
    plt.xlabel('Strain')
    plt.ylabel('Stress(Pa)')
    plt.title('Three Curves in the Second Plot')
    plt.legend()

    # 设置x轴和y轴为对数坐标
    plt.xscale('log')
    plt.yscale('log')
    x4_min =max(np.min(x4),1e-6)
    x4_max=np.max(x4)
    y4_min = max(np.min(y4),1e-6)
    y4_max = np.max(y4)
    # 设置x轴的范围，取数据范围的1/10和10倍
    plt.xlim(x4_min /10, x4_max * 10)
    # 设置y轴的范围，取数据范围的1/10和10倍
    plt.ylim(y4_min /10, y4_max * 10)
    # 调整子图布局
    plt.tight_layout()

    # 显示图形
    plt.show()


elif choice == "9":

    # 只读取文件名为数字的txt文件
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt') and f[:-4].isdigit()])



    dissipation_ratios = []
    strain_0_values = []

    # 遍历每个txt文件
    for file in files:
        file_path = os.path.join(folder_path, file)

        # 加载txt文件中的数据
        data = np.loadtxt(file_path)

        # 提取应力、应变和应变率
        stress = data[:, 1]  # 第二列
        strain = data[:, 2]  # 第三列

        # 计算耗散比率（stress对strain的积分）
        dissipation_ratio = (np.trapz(stress, strain))/(4*np.max(stress)*np.max(strain))
        dissipation_ratios.append(dissipation_ratio)


    # 存储所有找到的 Excel 文件路径
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否为 Excel 文件（可以根据实际情况调整）
            if file.endswith(".xlsx") or file.endswith(".xls"):
                if not file.startswith("~"):
                # 构建完整的文件路径
                    print(f"  find a excel: {file}, the file path is {root}.\n   Only need one excel, The find process is over.")
                    excel_file_path = os.path.join(root, file)
                    break

    df = pd.read_excel(excel_file_path)
    strain_0 = df['ɣ in -'].tolist()

    # 将strain_0存储
    strain_0_values = strain_0[:len(dissipation_ratios)]  # 保持数据对齐

    # 打印每一行的数据
    for strain_0, dissipation_ratio in zip(strain_0_values, dissipation_ratios):
        print(f"{strain_0} {dissipation_ratio}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(strain_0_values, dissipation_ratios, 'o-', label='Dissipation Ratio vs Strain_0')
    plt.xscale('log')  # 设置横坐标为对数坐标
    plt.xlabel('Strain')
    plt.ylabel('Dissipation Ratio')
    plt.title('Dissipation Ratio vs Strain_0')
    plt.legend()
    plt.grid(True)
    plt.show()


elif choice == "10":

    # 获取文件夹内所有Excel文件
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # 检查是否有唯一的Excel文件
    if len(excel_files) == 1:
        excel_file = os.path.join(folder_path, excel_files[0])

        # 读取Excel文件
        df = pd.read_excel(excel_file)

        # 读取G' in Pa和ɣ in -列
        G_prime = df["G' in Pa"].values
        strain_0 = df["ɣ in -"].values
    else:
        print("文件夹中没有唯一的Excel文件，或存在多个Excel文件。")
        exit()

    # 获取文件夹内所有数字命名的txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and f[:-4].isdigit()]

    # 定义存储所有计算出的G_cage的列表
    G_cage_values = []

    # 遍历每个txt文件
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)

        # 读取txt文件内容
        data = np.loadtxt(file_path)

        # 分别获取时间t，应力stress，应变strain，应变速率
        time = data[:, 0]
        stress = data[:, 1]
        strain = data[:, 2]
        strain_rate = data[:, 3]

        # 找到应力列最接近于0的值的索引n
        n = np.argmin(np.abs(stress))
        # 计算应力对应变的导数
        d_sigma_d_gamma = np.gradient(stress, strain)

        # 获取应力为0点的G_cage
        G_cage = d_sigma_d_gamma[n]


        G_cage_values.append(G_cage)  # 将G_cage值存储到列表中
        print(f"G_cage: {G_cage}")

    # 确保G_cage_values和G_prime长度一致
    if len(G_cage_values) != len(G_prime):
        print(f"G_cage的数量 ({len(G_cage_values)}) 和 G_prime ({len(G_prime)}) 数量不一致。")
        exit()

    # 计算G_cage / G_prime
    G_ratio_values = np.array(G_cage_values) / G_prime

    # 转换x轴数据为百分数
    x_values_percent = strain_0 * 100

    # 绘图
    plt.figure()
    plt.plot(x_values_percent, G_ratio_values, 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Strain (%)")
    plt.ylabel("G_cage / G'")
    plt.title("G_cage / G' vs Strain")
    plt.grid(True)
    plt.show()


elif choice == "11":


    # 定义文件夹路径
    folder_path = os.path.join(folder_path, 'Spp Data')
    save_path = os.path.join(folder_path, 'file_titles.txt')


    # 定义函数读取上次保存的文件编号和曲线标题
    def load_previous_selection(save_path):
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                file_numbers = []
                titles = []
                for line in lines:
                    parts = line.strip().split(', 曲线标题: ')
                    file_numbers.append(parts[0].split('文件编号: ')[1])
                    titles.append(parts[1])
                return file_numbers, titles
        return [], []


    # 询问用户是否要加载上次保存的文件编号和标题
    use_previous = input("是否加载上次保存的文件编号和曲线标题？（是请输入1/否请输入2）：")
    if use_previous.lower() == '1':
        file_numbers, titles = load_previous_selection(save_path)
        print(f"已加载文件编号: {file_numbers}")
        print(f"已加载曲线标题: {titles}")
    else:
        # 列出文件夹中已存在的txt文件
        existing_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if existing_files:
            print("Spp Data 文件夹中的txt文件有：")
            for file in existing_files:
                print(file)
        else:
            print("Spp Data 文件夹中没有找到txt文件。")

        # 用户输入文件编号（假设用户输入'01 02 03'）
        file_numbers = input("请输入文件编号（空格隔开）：").split()

        # 获取用户输入的曲线标题
        titles = []
        for num in file_numbers:
            title = input(f"请输入文件 {num}_spp data.txt 的曲线标题：")
            titles.append(title)

        # 询问是否保存文件编号和曲线标题
        save_data = input("是否保存文件编号和曲线标题？（是/否）：")
        if save_data.lower() == '是':
            with open(save_path, 'w', encoding='utf-8') as f:
                for num, title in zip(file_numbers, titles):
                    f.write(f"文件编号: {num}, 曲线标题: {title}\n")
            print(f"文件编号和曲线标题已保存到 {save_path}")

    # 定义不同的颜色渐变
    colormaps = [
        cm.winter,  # 橙黄-紫黑渐变
        cm.viridis,  # 蓝绿渐变
        cm.cool,  # 多色彩虹渐变（对比度较高）
        cm.Wistia,  # 蓝红渐变（强对比）
        cm.bwr,  # 全色相环
        cm.tab20,  # 离散颜色对比度强
        cm.twilight,  # 蓝紫-黄粉渐变
        cm.PuBuGn,  # 紫蓝绿渐变
        cm.YlGnBu,  # 黄绿蓝渐变
        cm.RdYlBu,  # 红黄蓝渐变（强对比）
        cm.plasma,  # 紫红渐变
        cm.nipy_spectral,  # 多彩强对比
        cm.Set1  # 强对比离散颜色
    ]


    #颜色1
    cmap1 = cm.get_cmap('autumn', 256)  # 第一个渐变颜色条
    cmap2 = cm.get_cmap('cool', 256)  # 第二个渐变颜色条
    #颜色2
    #cmap1 = cm.get_cmap('brg', 256)  # 第一个渐变颜色条
    #cmap2 = cm.get_cmap('jet', 256)  # 第二个渐变颜色条
    #颜色3
    #cmap1 = cm.get_cmap('Set1', 256)  # 第一个渐变颜色条
    #cmap2 = cm.get_cmap('jet', 256)  # 第二个渐变颜色条

    # 使用更高对比度和亮度的颜色条
    #cmap1 = cm.get_cmap('PuBu', 256)
    #cmap2 = cm.get_cmap('rainbow', 256)

    # 取颜色条的颜色，进行线性插值
    colors1 = cmap1(np.linspace(1, 0, 200))  # Spectral 颜色条从头到尾
    colors2 = cmap2(np.linspace(0.25, 1, 220))  # coolwarm 颜色条从头到尾

    # 拼接两个颜色条
    combined_colors = np.vstack((colors1, colors2))

    # 创建新的颜色条
    new_cmap = LinearSegmentedColormap.from_list('combined_cmap', combined_colors)

    # 将 colormap 设置为新的颜色条
    colormap = new_cmap
    # 初始化存储数据的字典，用于存储每个文件的曲线
    gt_prime_dict = {}
    gt_double_prime_dict = {}
    dgt_prime_dt_dict = {}
    dgt_double_prime_dt_dict = {}

    # 读取用户指定的文件
    for num in file_numbers:
        file_path = os.path.join(folder_path, f"{num}_spp data.txt")

        if os.path.exists(file_path):
            data = np.loadtxt(file_path)

            # 假设每个txt文件有四列数据
            gt_prime_dict[num] = data[:, 0]  # 第一列 Gt'
            gt_double_prime_dict[num] = data[:, 1]  # 第二列 Gt''
            dgt_prime_dt_dict[num] = data[:, 2]  # 第三列 dGt'/dt
            dgt_double_prime_dt_dict[num] = data[:, 3]  # 第四列 dGt''/dt
        else:
            print(f"文件 {num}_spp data.txt 不存在")

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # 绘制第一个散点图 (Gₜ' vs Gₜ'')
    for i, num in enumerate(file_numbers):
        if num in gt_prime_dict:
            Gt_prime = gt_prime_dict[num]
            Gt_double_prime = gt_double_prime_dict[num]

            # 从渐变中选取一个颜色，使用np.linspace来控制不同曲线的对比度
            color = colormap(i / len(file_numbers))  # 为每条曲线选择单一颜色

            # 绘制散点图，所有点使用同一个颜色
            axs[0].scatter(Gt_prime, Gt_double_prime, c=[color], s=50, alpha=0.8, edgecolors='w', linewidth=0.5)

    # 使用 Unicode 格式确保下标 t 正常显示
    axs[0].set_title('Cole–Cole plots of Gₜ\'(t) and Gₜ″(t)', fontweight='bold')
    axs[0].set_xlabel('Gₜ\'(t)', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Gₜ″(t)', fontsize=14, fontweight='bold')
    axs[0].grid(True)

    # 绘制第二个散点图 (dGₜ'/dt vs dGₜ''/dt)
    for i, num in enumerate(file_numbers):
        if num in dgt_prime_dt_dict:
            dGt_prime_dt = dgt_prime_dt_dict[num]
            dGt_double_prime_dt = dgt_double_prime_dt_dict[num]

            # 从渐变中选取一个颜色
            color = colormap(i / len(file_numbers))  # 为每条曲线选择单一颜色

            # 绘制散点图，所有点使用同一个颜色
            axs[1].scatter(dGt_prime_dt, dGt_double_prime_dt, c=[color], s=50, alpha=0.8, edgecolors='w', linewidth=0.5)

    # 使用 Unicode 格式确保下标 t 正常显示，并设置字体加粗
    axs[1].set_title('Cole–Cole plots of dGₜ\'(t)/dt and dGₜ″(t)/dt', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('dGₜ\'(t)/dt', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('dGₜ″(t)/dt', fontsize=14, fontweight='bold')
    axs[1].grid(True)

    # 创建图例
    legend_elements = []
    for i, title in enumerate(titles):
        color = colormap(i / len(file_numbers))

        # 添加颜色点和标题到图例元素中
        legend_elements.append(
            Line2D([0], [0], marker='o', color=color, label=title, markersize=10, linestyle='None'))

    # 显示图例
    axs[0].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, prop={'weight': 'bold'})
    axs[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, prop={'weight': 'bold'})
    # 使用 plt.setp 修改刻度标签字体为加粗
    plt.setp(axs[0].get_xticklabels(), fontweight='bold')
    plt.setp(axs[0].get_yticklabels(), fontweight='bold')
    plt.setp(axs[1].get_xticklabels(), fontweight='bold')
    plt.setp(axs[1].get_yticklabels(), fontweight='bold')

    file_name = 'Spp.png'
    # 完整文件路径
    save_path = os.path.join(folder_path, file_name)
    # 确保关闭所有子图的网格
    for ax in axs:
        ax.grid(False)
    # 自动调整布局
    plt.tight_layout()
    # 保存图片到指定路径，设置DPI为300
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # 保存图片时指定 DPI，确保图片清晰
    fig.savefig('your_figure_name.png', dpi=300)
    print("Spp 数据图片已保存")

    # 调整子图布局
    plt.tight_layout()

    # 去掉背景网格
    axs[0].grid(False)
    axs[1].grid(False)

    # 显示图形
    plt.show()


