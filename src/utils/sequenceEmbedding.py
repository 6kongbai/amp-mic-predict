import numpy as np  # 用于数值计算
from scipy import stats  # 用于科学计算，如此处的皮尔逊相关系数
from sklearn import metrics  # 用于模型评估，如计算MSE, MAE, R2
import os

# 获取当前代码文件所在目录（src/utils/）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def dictionary_substitution_matrix_features(filename):
    """
    从文件中读取氨基酸替换矩阵（如BLOSUM62）作为特征。
    参数:
        filename (str): 替换矩阵文件的相对路径（相对于 src/utils/features/）。
    返回:
        di_sub_mat_feat (dict): 一个字典，键是氨基酸，值是其对应的矩阵特征向量。
    """
    AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    di_sub_mat_feat = {}
    # 构建绝对路径
    file_path = os.path.join(BASE_DIR, "features", filename)
    with open(file_path) as f:
        for line in f:
            if line[0] in AAs:
                feats = line.split()[1:21]
                feats = list(map(np.float32, feats))
                di_sub_mat_feat[line[0]] = feats
    return di_sub_mat_feat


def AAindex_():
    """
    从AAindex数据文件加载氨基酸的物理化学性质。
    返回:
        AA_encoding (dict): 一个字典，键是氨基酸，值是其物理化学性质的向量。
    """
    # 构建绝对路径
    file_path = os.path.join(BASE_DIR, "features", "AAindex.txt")
    AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    with open(file_path) as f:
        records = f.readlines()[1:]  # 跳过标题行
    AAindex = []
    AA_encoding = {}
    for i in records:
        if i.rstrip():
            AAindex.append(i.rstrip().split()[1:])
    for i in range(20):
        AA_encoding[AAs[i]] = [float(item[i]) for item in AAindex]  # 确保转换为浮点数
    return AA_encoding


def dictionary_one_hot():
    """
    为20种标准氨基酸生成独热编码（One-Hot Encoding）。
    返回:
        di_one_hot (dict): 一个字典，键是氨基酸，值是其20维的独热编码向量。
    """
    AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    di_one_hot = {}
    for (i, aa) in enumerate(AAs):
        # 创建一个全零向量
        di_one_hot[aa] = np.zeros((len(AAs)), dtype=np.float32)
        # 将对应位置设为1
        di_one_hot[aa][i] = 1.0
    return di_one_hot


# Z-scale描述符：一种基于氨基酸物理化学性质（疏水性、大小、电荷等）的标准化特征
zscale = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30], 'C': [0.84, -1.67, 3.71, 0.18, -2.65],
    'D': [3.98, 0.93, 1.93, -2.46, 0.75], 'E': [3.11, 0.26, -0.11, -0.34, -0.25],
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62], 'G': [2.05, -4.06, 0.36, -0.82, -0.38],
    'H': [2.47, 1.95, 0.26, 3.90, 0.09], 'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
    'K': [2.29, 0.89, -2.49, 1.49, 0.31], 'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98], 'N': [3.05, 1.62, 1.04, -1.15, 1.61],
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00], 'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
    'R': [3.52, 2.50, -3.50, 1.99, -0.17], 'S': [2.39, -1.07, 1.15, -1.39, 0.67],
    'T': [0.75, -2.18, -1.12, -1.46, -0.40], 'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59], 'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
}


def seq_to_embedding(seq_list, output_vector, AA_len):
    """
    将氨基酸序列列表转换为固定长度的数值嵌入矩阵列表。
    参数:
        seq_list (list): 氨基酸序列的列表。
        output_vector (dict): 包含每种氨基酸特征向量的字典。
        AA_len (int): 序列的最大长度，用于填充或截断。
    返回:
        np.array: 一个三维数组 (序列数量, AA_len, 特征维度)。
    """
    output_list = list()
    for seq in seq_list:
        # 创建一个形状为 (序列长度, 特征维度) 的零矩阵
        output_array = np.zeros((AA_len, len(output_vector['A'])))
        for (index, aa) in enumerate(seq):
            if index < AA_len:  # 确保不超过最大长度
                output_array[index, :] = output_vector[aa]  # 查找并填充特征
        output_list.append(output_array)
    return np.array(output_list)


def evaluate(y_pred, y_test):
    """
    计算并打印回归模型的性能指标。
    参数:
        y_pred (array): 模型预测值。
        y_test (array): 真实值。
    返回:
        float: 皮尔逊相关系数 (PCC)。
    """
    MSE = metrics.mean_squared_error(y_test, y_pred)  # 均方误差
    MAE = metrics.mean_absolute_error(y_test, y_pred)  # 平均绝对误差
    R2 = metrics.r2_score(y_test, y_pred)  # R-squared (决定系数)
    PCC = stats.pearsonr(y_test, y_pred)  # 皮尔逊相关系数
    print('模型表现')
    print('MSE: {:0.3f}.'.format(MSE))
    print('MAE = {:0.3f}.'.format(MAE))
    print('R2 = {:0.3f}.'.format(R2))
    print('PCC = {:0.3f}.'.format(PCC[0]))

    return PCC[0]


def get_vector():
    """
    生成氨基酸特征向量的字典。
    注意：此函数在主脚本中未被调用。
    返回:
        output_vector (dict): 包含每种氨基酸的特征向量。
    """
    One_hot_encoding = dictionary_one_hot()
    BLOSUM62 = dictionary_substitution_matrix_features("BLOSUM62.txt")
    AAindex = AAindex_()  # 获取AAindex特征
    output_vector = {}  # 创建一个空字典用于存放组合特征
    AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for aa in AAs:
        # 将AAindex, BLOSUM62, One-hot, Z-scale四种特征向量拼接成一个长向量
        output_vector[aa] = np.concatenate((AAindex[aa], BLOSUM62[aa], One_hot_encoding[aa], zscale[aa]))
    return output_vector
