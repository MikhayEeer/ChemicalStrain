import os
import re

import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

class plotter(baseTask):
    def __init__(self, folder_path):
        super().__init__(folder_path)
        plt.rcParams['font.family'] = 'Times New Roman' 

    def exec_call(self):
        choice = select_choices()
        while choice not in ["0", "1", "2", "3", "4", 
                            "5", "6", "7", "8", "9", "10", "11"]:
            print("-- info -- 无效的选择, 请输入1-11之间的数字\n"
                  "-- info -- 请重新选择：")
            choice = select_choices()
        choice = int(choice)
        match choice:
            case 0:
                print("程序退出")
                return
            case 1:
                self.plot_lissajous()
            case 2:
                self.plot_stress_strain_normalization()
            case 3:
                self.plot_lissajous_overlay()
            case 4:
                self.plot_st_value()
            case 5:
                self.plot_same_strain_different_in_i1()
            case 6:
                self.plot_different_strain_same_in_i1()
            case 7:
                self.plot_spp_method()
            case 8:
                self.plot_stress_bifurcation()
            case 9:
                self.plot_dissipation_rate()
            case 10:
                self.plot_cage_modulus()
            case 11:
                self.plot_spp()         

    def select_choices(self):
        choice = input("请选择绘图选项:\n"
                       "0. 退出该程序\n"
                       "1. 绘制李萨如环\n"
                       "2. 应力应变归一化\n"
                       "3. 李萨如环叠加\n"
                       "4. ST值\n"
                       "5. 同一应变不同In/I1比较\n"
                       "6. 不同应变同一In/I1比较\n"
                       "7. SPP方法\n"
                       "8. 应力分叉\n"
                       "9. 耗散率\n"
                       "10. 笼子模量\n"
                       "11. Spp绘图程序(请确保运行'7'程序完成数据计算)\n"
                       "请输入您的选择: ")
        return choice

    def plot_lissajous(self, colors = "haline_r"):
        '''
        李萨如环
        '''
        file_name, file_number = self._select_file()
        data = self._load_data_from_txt(file_name)
        self._plot_lissajous_core([data], [file_number])

        
    def plot_stress_strain_normalization(self, 
                        output_folder_name="归一化结果"):
        '''
        应变应力归一化
        '''
        output_folder_path = self._create_output_folder(output_folder_name)

        for file_name in os.listdir(self.folder_path):
            if self.check_txt_digital_path(file_name):
                input_file_path = os.path.join(self.folder_path, file_name)
                output_file_path = os.path.join(output_folder_path,file_name)

                data = self._load_data_from_txt(input_file_path)

                max_stress = np.max(data[:,1])
                max_strain = np.max(data[:, 2])
                max_strain_rate = np.max(data[:, 3])

                data[:, 1] /= max_stress
                data[:, 2] /= max_strain
                data[:, 3] /= max_strain_rate

                np.savetxt(output_file_path, data, fmt='%f', delimiter='\t')  # 以制表符分隔保存数据

    def plot_lissajous_overlay(self):
        '''
        李萨如环叠加
        '''
        txt_files = self.get_numbers_txt_files()
        selected_indices_str = input("输入要选择的文件编号（空格分隔）：")
        selected_indices = [int(idx) - 1 for idx in selected_indices_str.split(' ')]
        selected_files = [txt_files[idx] for idx in selected_indices]

        data_list = []
        file_numbers = []
        for file_path in selected_files:
            data = self._load_data_from_txt(file_path)
            data_list.append(data)
            file_numbers.append(os.path.basename(file_path).split('.')[0])

        self._plot_lissajous_core(data_list, file_numbers)
    
    def plot_st_value(self):
        '''
        ST值
        '''
        strain_value = []
        excel_files = [file for file in os.listdir(self.folder_path) if file.endswith('.xlsx') or file.endswith('.xls')]
        for excel_file in excel_files:
            file_path = os.path.join(self.folder_path, excel_file)
            df = pd.read_excel(file_path)
            strain_column = df["ɣ in -"]
            strain_value.extend(strain_column)

        fourier_folder_path = os.path.join(self.folder_path, "fourier_coefficients")
        S_value = []
        T_value = []
        G_l_value=[]
        G_m_value=[]
        eta_l_value=[]
        eta_m_value = []
        for filename in os.listdir(fourier_folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(fourier_folder_path, filename)
                real = []
                imag = []
                with open(file_path, 'r') as file:
                    for line in file:
                        data = line.strip().split(',')
                        if len(data) >= 3:
                            real.append(float(data[1]))
                            imag.append(float(data[2]))
                    nl = 0
                    nm = 0
                    gl = 0
                    gm = 0
                    eta_l = 0
                    eta_m = 0
                    for i in range(0, len(real), 2):
                        nl += real[i]
                        nm += (i + 1) * real[i] * ((-1) ** (i / 2))
                        gl += (i + 1) * imag[i]
                        gm += imag[i] * ((-1) ** (i / 2))

                if gl != 0:
                    S = (gl - gm) / gl
                else:
                    S = 0

                if nl != 0:
                    T = (nl - nm) / nl  
                else:
                    T = 0
                    
                S_value.append(S)
                T_value.append(T)
                G_l_value.append(gl)
                G_m_value.append(gm)
                eta_l_value.append(nl)
                eta_m_value.append(nm)
        
        output_file_path = os.path.join(self.folder_path, "ST.txt")
        with open(output_file_path, 'w') as output_file:
            for S, T, G_l, G_m, eta_l, eta_m in zip(S_value, T_value,G_l_value,G_m_value,eta_l_value,eta_m_value):
                output_file.write(f"{S}\t{T}\t{G_l}\t{G_m}\t{eta_l}\t{eta_m}\n")

        print(f"结果已保存到 {output_file_path}")

        input_file_path = os.path.join(self.folder_path, "ST.txt")
        Gl, Gm, el, em = [], [], [], []

        with open(input_file_path, 'r') as input_file:
            for line in input_file:
                values = line.strip().split('\t')
                Gl.append(float(values[2]))
                Gm.append(float(values[3]))
                el.append(float(values[4]))
                
        mod_Gl = [g / s for g, s in zip(Gl, strain_value)]
        mod_Gm = [g / s for g, s in zip(Gm, strain_value)]
        mod_el = [e / (s*omega) for e, s in zip(el, strain_value)]
        mod_em = [e / (s*omega) for e, s in zip(em, strain_value)]

        output_mod_file_path = os.path.join(self.folder_path, "mod_ST.txt")
        with open(output_mod_file_path, 'w') as output_mod_file:
            for mg, mm, mel, mem in zip(mod_Gl, mod_Gm, mod_el, mod_em):
                output_mod_file.write(f"{mg}\t{mm}\t{mel}\t{mem}\n")

        print(f"结果已保存到 {output_mod_file_path}")

        self._setup_plot('Strain vs S', 'Stress', 'S', 10, 5, False)
        plt.scatter(strain_value, S_value, color='blue', marker='o', label='S')
        plt.xscale('log')
        plt.show()

        self._setup_plot('Strain vs T', 'Stress', 'T', 10, 5, False)
        plt.scatter(strain_value, T_value, color='red', marker='o', label='T')
        plt.xscale('log')
        plt.show()

    def plot_same_strain_different_in_i1(self):
        '''
        同一应变不同In/I1比较
        '''
        fourier_folder_path = os.path.join(self.folder_path, "fourier_coefficients")
        txt_files,_ = self.generate_txt_files_list(fourier_folder_path)

        def custom_sort(file_name):
            try:
                return int(file_name.split('.')[0])
            except ValueError:
                return float('inf')

        txt_files.sort(key=custom_sort)
                        
        excel_file_paths = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".xlsx") or file.endswith(".xls"):
                    excel_file_paths.append(os.path.join(root, file))

        df = pd.read_excel(excel_file_path)
        strain_values = df['ɣ in -'].tolist()

        for i, file_name in enumerate(txt_files):
            print(f"{i + 1}. {file_name}")

        selected_indices_str = input("输入要选择的文件编号（空格分隔）：")
        selected_indices = [int(idx) - 1 for idx in selected_indices_str.split(' ')]

        fig = go.Figure()
        file_data = []

        for i, idx in enumerate(selected_indices):
            if 0 <= idx < len(txt_files):
                file_name = txt_files[idx]
                file_path = os.path.join(F_folder_path, file_name)

                with open(file_path, 'r') as file:
                    lines = file.readlines()

                y_values = []

                for line in lines:
                    values = [float(value) for value in line.strip().split(',')]
                    y_values.append(values[0])

                x_values = list(range(1, len(y_values) + 1))

                max_y_value = max(y_values)
                y_values = [value / max_y_value for value in y_values]

                strain_value = strain_values[idx]
                file_data.append((file_name, x_values, [strain_value] * len(x_values), y_values))

                for file_name, x_values, y_values, z_values in file_data:
                    print(f"文件名: {file_name}")
                    print("Harmonic Strain In/I1:")
                    for x, y, z in zip(x_values, y_values, z_values):
                        print(f"{x}\t{y}\t{z}")
                    print()

                y_values_scaled = [y * 100 for y in y_values]

                fig.add_trace(go.Scatter3d(
                    x=x_values,
                    y=y_values_scaled,
                    z=z_values,
                    mode='lines',
                    line=dict(width=6),
                    name=file_name
                ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Harmonic', tickmode='auto'),
                yaxis=dict(
                    title='Strain (%)',
                    type="log",
                    tickmode='linear',
                    tickvals=[0.1, 1, 10, 100, 200, 500, 1000, 2000],
                    ticktext=["0.1","1",'10', '100',"200", '500', '1000',"2000"],
                    dtick=1,
                    tickformat=''
                ),
                zaxis=dict(
                    title='In/I1',
                    type="log",
                    tickmode='array',
                    tickvals=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                    ticktext=['10\u207B\u2075', '10\u207B\u2074', '10\u207B\u00B3', '10\u207B\u00B2', '10\u207B\u00B9',
                          '10\u2070'],
                    dtick=1
                )
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=1),
            title='3D 折线图',
        )

        fig.show()

    def plot_different_strain_same_in_i1(self):
        '''
        不同应变同一In/I1比较
        '''
        def calculate_ratio(file_path, n):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                first_line_data = float(lines[0].strip().split(",")[0])  # 读取第一行数据的第一列值
                nth_line_data = float(lines[n - 1].strip().split(",")[0])  # 读取第n行数据的第一列值
                ratio = nth_line_data / first_line_data
                return ratio


        F_folder_path = os.path.join(self.folder_path, "fourier_coefficients")

        # 存储所有找到的 Excel 文件路径
        excel_file_paths = []

        # 遍历文件夹
        for root, dirs, files in os.walk(self.folder_path):
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
        
    def plot_spp_method(self):
        '''
        SPP方法
        '''
        om = 2 * np.pi

        file_name, file_number = self._select_file('fourier_coefficients')

        with open(file_name, 'r') as file:
            lines = file.readlines()

        N = int(input("请输入N: "))

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
        input_file_path = os.path.join(self.folder_path, f"{file_number}.txt")

        with open(input_file_path, 'r') as input_file:
            data = input_file.readlines()

        # 提取第二列数据
        stress_values_old = [line.split()[1] for line in data]
        strain_values = [line.split()[2] for line in data]
        strain_rate_values = [line.split()[3] for line in data]



        # 创建 "rebuild" 文件夹
        rebuild_folder_path = os.path.join(self.folder_path, "rebuild")
        os.makedirs(rebuild_folder_path, exist_ok=True)

        # 写入新的 txt 文件，包括第二列数据和 r(t) 值
        output_file_path = os.path.join(rebuild_folder_path, f"{file_number}_rebuild.txt")
        with open(output_file_path, 'w') as output_file:
            for i in range(len(strain_values)):
                output_file.write(f"{t[i]}\t{stress_values[i]}\t{strain_values[i]}\t{strain_rate_values[i]}\n")

        data_new = np.loadtxt(output_file_path)

        data_read_num =256
        t =data_new[:data_read_num,0]

        stress = data_new[:data_read_num, 1]
        strain = data_new[:data_read_num, 2]

        # 读取 strain_rate 列的前一半行
        strain_rate = data_new[:data_read_num, 3] / om
        # 构建向量P
        P = np.column_stack((strain, strain_rate, stress))

        # 计算P对时间的导数，即P'(t)
        P_prime = np.gradient(P, t, axis=0)

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
        rebuild_folder_path_2 = os.path.join(self.folder_path, "SPP Data")
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

    def plot_stress_bifurcation(self):
        '''
        应力分叉
        '''
        max_a_values = []
        max_b_values = []
        max_c_values = []
        d_values = []
        e_values = []
        f_values = []
        g_values = []


        # 遍历文件夹中的所有txt文件
        for filename in os.listdir(self.folder_path):
            if re.match(r'^\d+\.txt$', filename):
                file_path = os.path.join(self.folder_path, filename)

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
        output_file_path = routput_file_path = os.path.join(self.folder_path, "应力分叉.txt")
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

    def plot_dissipation_rate(self):
        '''
        耗散率
        '''
        # 只读取文件名为数字的txt文件
        txt_files = self.get_numbers_txt_files()
        txt_files = sorted(txt_files)

        dissipation_ratios = []
        strain_0_values = []

        # 遍历每个txt文件
        for file in txt_files:
            file_path = os.path.join(self.folder_path, file)

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
        for root, dirs, files in os.walk(self.folder_path):
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
    
    def plot_cage_modulus(self):
        '''
        笼子模型
        '''
        pass

    def plot_spp():
        '''
        spp绘图 , 需要choice 7进行数据计算
        '''
        pass

# private:
    def _plot_lissajous_core(self, data_list, file_numbers, color="haline_r"):
        '''
        李萨如环核心绘制函数
        '''
        fig_3d = go.Figure()
        fig_projections = make_subplots(rows=1, cols=3,
                                        subplot_titles=['Elastic Lissajous Curve',
                                                        'Viscous Lissajous Curve',
                                                        'Z-axis Projection'])

        for data, file_number in zip(data_list, file_numbers):
            stress = data[:, 1]
            strain = data[:, 2]
            strain_rate = data[:, 3]

            # 3D图
            fig_3d.add_trace(go.Scatter3d(x=strain, y=strain_rate, z=stress, mode='markers',
                                          marker=dict(size=6, color=stress, colorscale=colors, opacity=0.8),
                                          name=f'File {file_number}'))

            # 投影图
            fig_projections.add_trace(go.Scatter(x=strain, y=stress, mode='markers',
                                                 marker=dict(size=12, color=stress, colorscale=colors, opacity=0.8),
                                                 name=f'File {file_number} (Elastic)'), row=1, col=1)
            fig_projections.add_trace(go.Scatter(x=strain_rate, y=stress, mode='markers',
                                                 marker=dict(size=12, color=stress, colorscale=colors, opacity=0.8),
                                                 name=f'File {file_number} (Viscous)'), row=1, col=2)
            fig_projections.add_trace(go.Scatter(x=strain, y=strain_rate, mode='markers',
                                                 marker=dict(size=12, color=stress, colorscale=colors, opacity=0.8),
                                                 name=f'File {file_number} (Z-axis)'), row=1, col=3)

        # 更新3D图布局
        fig_3d.update_layout(scene=dict(xaxis_title='Strain',
                                        yaxis_title='Strain Rate',
                                        zaxis_title='Stress',
                                        aspectmode='cube'),
                             title='Stress-Strain Rate-Strain Relationship')

        # 更新投影图布局
        fig_projections.update_layout(height=600, width=1500,
                                      title='Stress-Strain Rate-Strain Projections')

        # 显示图形
        pio.show(fig_3d)
        pio.show(fig_projections)        

    def _select_file(self, subfolder_name = ''):
        file_number = input("请输入文件编号：")
        if subfolder_name=='':
            file_name = os.path.join(self.folder_path, f"{file_number}.txt")
        else:
            file_name = os.path.join(self.folder_path, subfolder_name, f"{file_number}.txt")
        
        while not os.path.exists(file_name):
            file_number = input("文件不存在，请重新输入文件编号：")
            if subfolder_name=='':
                file_name = os.path.join(self.folder_path, f"{file_number}.txt")
            else:
                file_name = os.path.join(self.folder_path, subfolder_name, f"{file_number}.txt")
        
        return file_name, file_number
    
    def _load_data_from_txt(self, txt_file_path):
        return np.loadtxt(txt_file_path)
    
    def _create_output_folder(self, output_folder_name):
        output_folder_path = os.path.join(self.folder_path, 
                                            output_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        return output_folder_path
    
    def _setup_plot(self, title, xlabel, ylabel, h=10, w=4, grid=True):
        plt.figure(figsize=(h, w))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
        
        