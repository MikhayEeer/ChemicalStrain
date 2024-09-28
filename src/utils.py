# utility functions
import numpy as np


class MathUtils:

    def __init__(self, txt_files):
        self.txt_files = txt_files

    def sin_func(x, A, omega, phi, C):
        return A * np.sin(omega * x + phi) + C

    def derivative(x, A, omega, phi, C):
        return omega * A  * np.cos(omega * x + phi)

    # 得到剪切速率值
    def get_shear_rate(self):
        from scipy.optimize import curve_fit

        for txt_file in self.txt_files:
            data = np.loadtxt(txt_file)
            x = data[:, 0]/512
            y = data[:, 2]
            
            A0 = (np.max(y) - np.min(y)) / 2
            omega0 = 2 * np.pi / np.max(x)

            params, covariance = curve_fit(self.sin_func, x, y, p0=[A0, omega0, 0, 0])
            A, omega, phi, C = params

            new_y = self.sin_func(x, A, omega, phi, C)
            data[:, 2] = new_y

            derivatives = self.derivative(x, A, omega, phi, C)

            data_with_derivatives = np.column_stack((data, derivatives))

            print(f"##debug## 剪切速率值--保存文件: {txt_file}")
            np.savetxt(txt_file, data_with_derivatives, delimiter='\t')

    def fourier_transform(self):
        #傅里叶变换
        output_folder = os.path.join(folder_path, "fourier_coefficients")

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        for txt_file in self.txt_files:
            with open(txt_file, 'r') as file:
                lines = file.readlines()
            data = []
            strain = []
            for line in lines:
                columns = line.split()
                if len(columns) >= 3:
                    data.append(float(columns[1]))
                    strain.append(float(columns[2]))
            
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

            output_filepath = os.path.join(self.folder_path, os.path.basename(txt_file))
    
            with open(output_filepath, 'w') as output_file:
                for n in range(1, 15):
                    I_n, real_part, imag_part = result[n-1]
                    output_file.write(f"{I_n}\t{real_part}\t{imag_part}\n")

            print(f"##debug## 傅里叶变换--保存文件: {output_filepath}")
