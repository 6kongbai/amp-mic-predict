import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理和CSV文件I/O
import collections  # 提供额外的数据结构
from sklearn.model_selection import train_test_split  # 用于划分数据集
import tensorflow as tf  # 主要的深度学习框架
import tensorflow.keras as keras  # TensorFlow的高级API，用于构建神经网络
import sklearn as sk  # 机器学习库
from sklearn import metrics  # 用于模型评估，如计算MSE, MAE, R2
from scipy import stats  # 用于科学计算，如此处的皮尔逊相关系数
import random  # 随机数生成
import pickle  # 用于序列化和反序列化Python对象
from tensorflow.keras.models import load_model  # 用于加载预训练的Keras模型
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Dropout, Flatten, GlobalAveragePooling1D, LSTM, \
    Conv1D  # Keras中的各种层
from tensorflow.keras.layers import LayerNormalization  # 层归一化
from tensorflow.keras.models import Model  # Keras模型类
import tensorflow.keras.backend as K  # Keras后端，用于底层操作

# 注意：以下两个函数依赖于未在此处导入的库。
# 需要安装：pip install biopython gensim
from Bio import SeqIO  # 用于解析FASTA文件
from gensim.models import Word2Vec  # 用于训练Word2Vec模型


# --- 特征工程函数 ---

def dictionary_word2vec(filename):
    """
    从FASTA文件生成氨基酸的Word2Vec嵌入。
    注意：此函数在主脚本中未被调用。
    参数:
        filename (str): FASTA文件的路径。
    返回:
        di_word2vec (dict): 一个字典，键是氨基酸，值是其Word2Vec向量。
    """
    di_word2vec = {}
    fasta_file = filename
    seq_list = []
    # 使用Biopython的SeqIO解析FASTA文件，提取所有序列
    for (index, seq_record) in enumerate(SeqIO.parse(fasta_file, "fasta")):
        seq_list.append(str(seq_record.seq))
    arr = np.array(seq_list)
    # 使用gensim的Word2Vec模型训练嵌入，向量维度为20
    w2v_model = Word2Vec(arr, vector_size=20)
    # 将训练好的嵌入存入字典
    for idx, key in enumerate(w2v_model.wv.key_to_index):
        di_word2vec[key] = list(w2v_model.wv[key])
    return di_word2vec


def dictionary_substitution_matrix_features(filename):
    """
    从文件中读取氨基酸替换矩阵（如BLOSUM62）作为特征。
    参数:
        filename (str): 替换矩阵文件的路径。
    返回:
        di_sub_mat_feat (dict): 一个字典，键是氨基酸，值是其对应的矩阵特征向量。
    """
    AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y",
           "V"]  # 20种标准氨基酸
    di_sub_mat_feat = {}
    for line in open(filename):
        # 如果行的第一个字符是一种氨基酸
        if (line[0] in AAs):
            # 提取与20种标准氨基酸对应的20个特征值
            feats = line.split()[1:21]
            feats = list(map(np.float32, feats))  # 转换为浮点数
            di_sub_mat_feat[line[0]] = feats  # 存入字典
    return di_sub_mat_feat


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


def AAindex():
    """
    从AAindex数据文件加载氨基酸的物理化学性质。
    返回:
        AA_encoding (dict): 一个字典，键是氨基酸，值是其物理化学性质的向量。
    """
    filename = "features/AAindex.txt"  # AAindex数据文件路径
    AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    with open(filename) as f:
        records = f.readlines()[1:]  # 跳过标题行
    AAindex = []
    AA_encoding = {}
    # 解析文件内容
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
    # 为每种氨基酸构建其特征向量
    for i in range(20):
        AA_encoding[AAs[i]] = [item[i] for item in AAindex]
    return AA_encoding


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


# --- 模型评估和工具函数 ---

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


def train_test_val_DL(train, test, val):
    """
    从DataFrame中分离出特征（序列）和标签（浓度）。
    参数:
        train, test, val (pd.DataFrame): 训练、测试、验证集的DataFrame。
    返回:
        元组: 包含6个元素的元组 (train_x, test_x, val_x, train_y, test_y, val_y)。
    """
    train_x = train['SEQUENCE']
    train_y = train['NEW-CONCENTRATION']
    test_x = test['SEQUENCE']
    test_y = test['NEW-CONCENTRATION']
    val_x = val['SEQUENCE']
    val_y = val['NEW-CONCENTRATION']
    return train_x, test_x, val_x, train_y, test_y, val_y


def combine_features(X1, X2):
    """
    水平堆叠（拼接）两个特征数组。
    注意：此函数在主脚本中未被调用。
    """
    combine_list = list()
    for i in range(len(X1)):
        combine_list.append(np.hstack((X1[i], X2[i])))
    return np.array(combine_list)


def save_model_history(model, history, model_name):
    """
    保存Keras模型及其训练历史。
    注意：此函数在主脚本中未被调用。
    """
    model.save('{}.h5'.format(model_name))  # 保存模型为HDF5文件
    hist_df = pd.DataFrame(history.history)  # 将历史记录转换为DataFrame
    hist_json_file = '{}_history.json.'.format(model_name)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)  # 保存历史记录为JSON文件


def Genomesequence_concat(Feature_array, Genome_array):
    """
    将序列特征（如T5嵌入）和基因组特征拼接在一起。
    参数:
        Feature_array (np.array): 序列特征数组 (e.g., shape: [samples, 40, 1024])。
        Genome_array (np.array): 基因组特征数组 (e.g., shape: [samples, 84])。
    返回:
        np.array: 拼接后的数组 (shape: [samples, 41, 1024])。
    """
    # 为基因组数据增加一个维度，使其变为 (samples, 1, features)
    arr = np.expand_dims(Genome_array, axis=1)
    # 定义填充后的目标形状
    pad_shape = (Feature_array.shape[0], 1, Feature_array.shape[2])
    # 在最后一个维度（特征维度）上用零填充基因组数据，使其与序列特征的维度匹配
    arr_padded = np.pad(arr, [(0, 0), (0, 0), (0, pad_shape[2] - arr.shape[2])], mode='constant')
    # 沿着第二个维度（序列长度维度）将两个数组拼接起来
    concatenated_array = np.concatenate((Feature_array, arr_padded), axis=1)
    return concatenated_array


def r_squared(y_true, y_pred):
    """
    自定义的R-squared指标，用于Keras模型编译。
    这是为了在加载使用此自定义指标训练的模型时不会出错。
    """
    ss_res = K.sum(K.square(y_true - y_pred))  # 残差平方和
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # 总平方和
    return 1 - ss_res / (ss_tot + K.epsilon())  # K.epsilon() 防止除以零


# --- 数据准备 ---

# 1. 生成并组合多种手工设计的氨基酸特征
One_hot_encoding = dictionary_one_hot()
BLOSUM62 = dictionary_substitution_matrix_features("features/BLOSUM62.txt")
AAindex = AAindex()
output_vector = {}  # 创建一个空字典用于存放组合特征
AAs = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
for aa in AAs:
    # 将AAindex, BLOSUM62, One-hot, Z-scale四种特征向量拼接成一个长向量
    output_vector[aa] = np.concatenate((AAindex[aa], BLOSUM62[aa], One_hot_encoding[aa], zscale[aa]))

# 2. 加载数据集 (序列长度最大为40)
# 大肠杆菌 (EC)
EC_train = pd.read_csv('data/EC_X_train_40.csv')
EC_test = pd.read_csv('data/EC_X_test_40.csv')
EC_val = pd.read_csv('data/EC_X_val_40.csv')

# 金黄色葡萄球菌 (SA)
SA_train = pd.read_csv('data/SA_X_train_40.csv')
SA_test = pd.read_csv('data/SA_X_test_40.csv')
SA_val = pd.read_csv('data/SA_X_val_40.csv')

# 铜绿假单胞菌 (PA)
PA_train = pd.read_csv('data/PA_X_train_40.csv')
PA_test = pd.read_csv('data/PA_X_test_40.csv')
PA_val = pd.read_csv('data/PA_X_val_40.csv')

# 3. 分离特征和标签
SA_X_train, SA_X_test, SA_X_val, SA_y_train, SA_y_test, SA_y_val = train_test_val_DL(SA_train, SA_test, SA_val)
EC_X_train, EC_X_test, EC_X_val, EC_y_train, EC_y_test, EC_y_val = train_test_val_DL(EC_train, EC_test, EC_val)
PA_X_train, PA_X_test, PA_X_val, PA_y_train, PA_y_test, PA_y_val = train_test_val_DL(PA_train, PA_test, PA_val)

# 4. 将氨基酸序列转换为基于手工特征的嵌入矩阵
# 注意：这部分嵌入数据在后续代码中没有被直接用于模型预测，但可能是为了对比或早期实验
SA_X_train = seq_to_embedding(SA_X_train, output_vector, 40)
SA_X_test = seq_to_embedding(SA_X_test, output_vector, 40)
SA_X_val = seq_to_embedding(SA_X_val, output_vector, 40)
# (对EC和PA数据集执行同样操作)
EC_X_train = seq_to_embedding(EC_X_train, output_vector, 40)
EC_X_test = seq_to_embedding(EC_X_test, output_vector, 40)
EC_X_val = seq_to_embedding(EC_X_val, output_vector, 40)

PA_X_train = seq_to_embedding(PA_X_train, output_vector, 40)
PA_X_test = seq_to_embedding(PA_X_test, output_vector, 40)
PA_X_val = seq_to_embedding(PA_X_val, output_vector, 40)

# 5. 加载预先计算好的T5蛋白质语言模型嵌入
# T5嵌入提供了更丰富的上下文信息，通常比手工特征效果更好
# EC
T5XL_EC_X_train = np.load('data/T5XL_Embeddings_max_40/EC_X_TRAIN.npy')
T5XL_EC_X_test = np.load('data/T5XL_Embeddings_max_40/EC_X_TEST.npy')
T5XL_EC_X_val = np.load('data/T5XL_Embeddings_max_40/EC_X_VAL.npy')
# SA
T5XL_SA_X_train = np.load('data/T5XL_Embeddings_max_40/SA_X_TRAIN.npy')
T5XL_SA_X_test = np.load('data/T5XL_Embeddings_max_40/SA_X_TEST.npy')
T5XL_SA_X_val = np.load('data/T5XL_Embeddings_max_40/SA_X_VAL.npy')
# PA
T5XL_PA_X_train = np.load('data/T5XL_Embeddings_max_40/PA_X_TRAIN.npy')
T5XL_PA_X_test = np.load('data/T5XL_Embeddings_max_40/PA_X_TEST.npy')
T5XL_PA_X_val = np.load('data/T5XL_Embeddings_max_40/PA_X_VAL.npy')

# 6. 将T5嵌入与基因组特征拼接
# 从原始DataFrame中提取基因组特征列 (iloc[:,250:-12])
# EC
T5XL_EC_X_train_GS = Genomesequence_concat(T5XL_EC_X_train, EC_train.iloc[:, 250:-12])
T5XL_EC_X_test_GS = Genomesequence_concat(T5XL_EC_X_test, EC_test.iloc[:, 250:-12])
T5XL_EC_X_val_GS = Genomesequence_concat(T5XL_EC_X_val, EC_val.iloc[:, 250:-12])
# SA
T5XL_SA_X_train_GS = Genomesequence_concat(T5XL_SA_X_train, SA_train.iloc[:, 250:-12])
T5XL_SA_X_test_GS = Genomesequence_concat(T5XL_SA_X_test, SA_test.iloc[:, 250:-12])
T5XL_SA_X_val_GS = Genomesequence_concat(T5XL_SA_X_val, SA_val.iloc[:, 250:-12])
# PA
T5XL_PA_X_train_GS = Genomesequence_concat(T5XL_PA_X_train, PA_train.iloc[:, 250:-12])
T5XL_PA_X_test_GS = Genomesequence_concat(T5XL_PA_X_test, PA_test.iloc[:, 250:-12])
T5XL_PA_X_val_GS = Genomesequence_concat(T5XL_PA_X_val, PA_val.iloc[:, 250:-12])

# --- 模型加载 ---
# 加载大量预训练好的模型文件
# 这些模型似乎是针对不同特征和架构组合进行训练的

# 模型组1: 基于手工特征的模型 (脚本后续部分未使用)
EC_bilstm_40 = load_model('model_max_40/EC_bilstm_40.h5')
# ... 其他类似模型

# 模型组2: 基于T5嵌入的模型 (脚本后续部分未使用)
T5_EC_bilstm_40 = load_model('model_max_40/T5_EC_bilstm_40.h5')
# ... 其他类似模型

# 模型组3: 基于T5嵌入+基因组特征的组合模型 (这是脚本核心使用的模型)
T5_Three_Bi_model = load_model('model_max_40/T5_Three_Bi_40.h5')  # BiLSTM模型
T5_Three_CNN_model = load_model('model_max_40/T5_Three_CNN_40.h5')  # CNN模型
T5_Three_Tf_model = load_model('model_max_40/T5_Three_Tf_40.h5')  # Transformer模型
# 加载多分支模型(MB)，需要指定自定义的r_squared指标
T5_Three_MB_model = load_model('model_max_40/T5_Three_MB_40.h5', custom_objects={'r_squared': r_squared})

# --- 模型评估与集成 ---

# 1. 准备用于存储预测结果的DataFrame
SA_pred = pd.DataFrame(columns={'CNN', 'BILSTM', 'MB'})
EC_pred = pd.DataFrame(columns={'CNN', 'BILSTM', 'MB'})
PA_pred = pd.DataFrame(columns={'CNN', 'BILSTM', 'MB'})

# 2. 将测试数据和标签整理到字典中，方便循环处理
X_test = {'SA_X_test': T5XL_SA_X_test_GS,
          'EC_X_test': T5XL_EC_X_test_GS,
          'PA_X_test': T5XL_PA_X_test_GS}
y_test = {'SA_y_test': SA_test['NEW-CONCENTRATION'],
          'EC_y_test': EC_test['NEW-CONCENTRATION'],
          'PA_y_test': PA_test['NEW-CONCENTRATION']
          }
pred = {'SA_pred': SA_pred,
        'EC_pred': EC_pred,
        'PA_pred': PA_pred}

# 3. 循环评估每个模型在三个细菌测试集上的表现
print('评估 T5_Three_CNN 模型')
# 遍厉三个测试集 (SA, EC, PA)
for X, y, p in zip(X_test, y_test, pred):
    print(X, y)  # 打印当前正在处理的数据集名称
    # 进行预测。模型输入需要将序列特征和基因组特征分开
    # X_test.get(X)[:,:40,:] 是T5序列嵌入部分 (前40行)
    # X_test.get(X)[:,40,:][:,:84] 是基因组特征部分 (第41行，并取前84列)
    CNN_pred = T5_Three_CNN_model.predict([X_test.get(X)[:, :40, :], X_test.get(X)[:, 40, :][:, :84]])
    # 评估预测结果
    CNN_PCC = evaluate(CNN_pred.reshape(-1), y_test.get(y))
    # 将预测结果存入对应的DataFrame
    pred.get(p)['CNN'] = CNN_pred.reshape(-1).tolist()
print('---------------------------------------')

print('评估 T5_Three_Bi (BiLSTM) 模型')
for X, y, p in zip(X_test, y_test, pred):
    print(X, y)
    Bi_pred = T5_Three_Bi_model.predict([X_test.get(X)[:, :40, :], X_test.get(X)[:, 40, :][:, :84]])
    Bi_PCC = evaluate(Bi_pred.reshape(-1), y_test.get(y))
    pred.get(p)['BILSTM'] = Bi_pred.reshape(-1).tolist()
print('---------------------------------------')

print('评估 T5_Three_Transformer 模型')
for X, y in zip(X_test, y_test):
    print(X, y)
    Tf_pred = T5_Three_Tf_model.predict([X_test.get(X)[:, :40, :], X_test.get(X)[:, 40, :][:, :84]])
    Tf_PCC = evaluate(Tf_pred.reshape(-1), y_test.get(y))
    # 注意：Transformer模型的预测结果没有被存储用于后续集成
print('---------------------------------------')

print('评估 T5_Three_MB (Multi-Branch) 模型')
for X, y, p in zip(X_test, y_test, pred):
    print(X, y)
    # 多分支模型的输入可能更复杂，这里输入了两次序列特征和一次基因组特征
    MB_40_pred = T5_Three_MB_model.predict(
        [X_test.get(X)[:, :40, :], X_test.get(X)[:, :40, :], X_test.get(X)[:, 40, :][:, :84]])
    MB_40_PCC = evaluate(MB_40_pred.reshape(-1), y_test.get(y))
    pred.get(p)['MB'] = MB_40_pred.reshape(-1).tolist()

# 4. 集成学习：对CNN, BILSTM, MB三个模型的预测结果进行加权平均
# 权重分配：CNN 30%, BiLSTM 40%, MB 30%
SA_pred['MIC_Final'] = SA_pred['CNN'] * 0.3 + SA_pred['BILSTM'] * 0.4 + SA_pred['MB'] * 0.3
EC_pred['MIC_Final'] = EC_pred['CNN'] * 0.3 + EC_pred['BILSTM'] * 0.4 + EC_pred['MB'] * 0.3
PA_pred['MIC_Final'] = PA_pred['CNN'] * 0.3 + PA_pred['BILSTM'] * 0.4 + PA_pred['MB'] * 0.3

# 最终的预测结果存储在 SA_pred, EC_pred, PA_pred 这三个DataFrame的'MIC_Final'列中
