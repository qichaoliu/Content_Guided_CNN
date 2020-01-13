import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing


# ratio: choose training\validation\testing sample according to "ratio" or "same number per class"
samples_type=['ratio','same_num'][0] #

for (curr_train_ratio,FLAG) in [(0.01,4)]:#(0.05,5),(0.01,4),(0.05,5),(0.01,2)
    OA_ALL,AA_ALL,KPP_ALL,AVG_ALL=[],[],[],[]
    
    for curr_seed in [0,1,2,3,4,]:
        if FLAG == 1:
            data_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_corrected.mat')
            data = data_mat['indian_pines_corrected']
            gt_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_gt.mat')
            gt = gt_mat['indian_pines_gt']

            # train_ratio = 0.1  # training ratio
            val_ratio = 0.01  # validation ratio
            class_count = 16
            learning_rate = 5e-4  # leaning rate of parameters
            learning_rate_sigma = 0.005 # leaning rate of sigma
            max_epoch = 800
            dataset_name = "indian_BF"
        
            # cut image into saveral cubes
            split_height = 1
            split_width = 1
            
            First_Chanels = 128
            After_Chanels = 32
            SIGMA = 1
            pass
        if FLAG == 2:
            data_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\PaviaU.mat')
            data = data_mat['paviaU']
            gt_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\Pavia_University_gt.mat')
            gt = gt_mat['pavia_university_gt']
            
            # 参数预设
            # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 9
            learning_rate = 5e-4
            learning_rate_sigma = 0.005
            max_epoch = 800
            dataset_name = "paviaU_BF"  # 数据集名称
            # 定义图像的切分块数，高分为2段，宽分为2段
            split_height = 3
            split_width = 1
            First_Chanels = 128
            After_Chanels = 32
            SIGMA = 1
            
            pass
        if FLAG == 3:
            data_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_corrected.mat')
            data = data_mat['salinas_corrected']
            gt_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_gt.mat')
            gt = gt_mat['salinas_gt']
            
            # 参数预设
            train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 16  # 样本类别数
            learning_rate = 1e-3  # 学习率
            learning_rate_sigma = 0.005
            max_epoch = 500  # 迭代次数
            dataset_name = "salinas_BF"  # 数据集名称
            split_height = 3
            split_width = 1
            First_Chanels = 128
            After_Chanels = 32
            
            SIGMA = 1
            pass
        if FLAG == 4:
            data_mat = sio.loadmat('..\\HyperImage_data\\Simu\\Simu_data.mat')
            data = data_mat['Simu_data']
            gt_mat = sio.loadmat('..\\HyperImage_data\\Simu\\Simu_label.mat')
            gt = gt_mat['Simu_label']
            
            # 参数预设
            # train_ratio = 0.004  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 5  # 样本类别数
            learning_rate = 5e-4  # 学习率
            learning_rate_sigma = 0.005
            max_epoch = 800  # 迭代次数
            dataset_name = "simu_BF"  # 数据集名称
            split_height = 1
            split_width = 1
            First_Chanels = 128
            After_Chanels = 32
            SIGMA = 1
            pass
        if FLAG == 5:
            data_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC.mat')
            data = data_mat['KSC']
            gt_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC_gt.mat')
            gt = gt_mat['KSC_gt']
            
            # 参数预设
            # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 13  # 样本类别数
            learning_rate = 5e-4  # 学习率
            learning_rate_sigma = 0.005
            max_epoch = 500  # 迭代次数
            dataset_name = "KSC_BF"  # 数据集名称
            split_height = 2
            split_width = 3
            First_Chanels = 128
            After_Chanels = 32
            SIGMA = 1
            pass
        if FLAG == 6:
            data_mat = sio.loadmat('..\\HyperImage_data\\paviaC\\paviaC.mat')
            data = data_mat['resultNew']
            gt_mat = sio.loadmat('..\\HyperImage_data\\paviaC\\pavia_center_gt.mat')
            gt = gt_mat['pavia_center_label']
            # 参数预设
            train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 9  # 样本类别数
            learning_rate = 5e-4  # 学习率
            learning_rate_sigma = 0.005
            max_epoch = 800  # 迭代次数
            dataset_name = "paviaC_BF"  # 数据集名称
            split_height = 3
            split_width = 3
            First_Chanels = 128
            After_Chanels = 32
            SIGMA = 1
            pass
        if FLAG == 7:
            data_mat = sio.loadmat('..\\HyperImage_data\\Botswana\\Botswana.mat')
            data = data_mat['Botswana']
            gt_mat = sio.loadmat('..\\HyperImage_data\\Botswana\\Botswana_gt.mat')
            gt = gt_mat['Botswana_gt']
            
            # 参数预设
            train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 14  # 样本类别数
            learning_rate = 1e-3  # 学习率
            learning_rate_sigma = 0.01
            max_epoch = 200  # 迭代次数
            dataset_name = "Botswana_BF"  # 数据集名称
            split_height = 6
            split_width = 1
            First_Chanels = 128
            After_Chanels = 32
            SIGMA = 0.2
            pass
        if FLAG == 8:
            data_mat = sio.loadmat('..\\HyperImage_data\\WDC\\WDC.mat')
            data = data_mat['wdc']
            gt_mat = sio.loadmat('..\\HyperImage_data\\WDC\\WDC_gt.mat')
            gt = gt_mat['wdc_gt']
            
            # 参数预设
            train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 7  # 样本类别数
            learning_rate = 5e-4  # 学习率
            learning_rate_sigma = 0.005
            max_epoch = 200  # 迭代次数
            dataset_name = "WDC_BF"  # 数据集名称
            split_height = 6
            split_width = 1
            First_Chanels = 128
            After_Chanels = 32
            SIGMA = 1
            pass

        #if "samples_type"="same_num" then "curr_train_ratio" denotes the number of training samples perclass
        train_samples_per_class = curr_train_ratio

        # if "samples_type"="same_num" then "val_samples" equals to 1 sample perclass
        val_samples = class_count
        
        # define the size of guided-kernel
        kernelsize = 5
        
        train_ratio=curr_train_ratio
        if split_height == split_width == 1:
            EDGE = 0
        else:
            EDGE = 10
        
        data = np.array(data, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        
        
        def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
            '''
            get classification map , then save to given path
            :param label: classification label, 2D
            :param name: saving path and file's name
            :param scale: scale of image. If equals to 1, then saving-size is just the label-size
            :param dpi: default is OK
            :return: null
            '''
            fig, ax = plt.subplots()
            numlabel = np.array(label)
            v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
            ax.set_axis_off()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
            foo_fig = plt.gcf()  # 'get current figure'
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
            pass
        
        
        def Save_Image(image, name, scale=4.0, dpi: int = 400):
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            fig.set_size_inches(image.shape[1] * scale / dpi, image.shape[0] * scale / dpi)
            foo_fig = plt.gcf()  # 'get current figure'
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.imshow(image)
            foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
            pass
        
        
        cmap = cm.get_cmap('jet', class_count + 1)
        plt.set_cmap(cmap)
        height, width, bands = data.shape  # 原始高光谱数据的三个维度
        m, n, d = data.shape  # 高光谱数据的三个维度
        
        # 数据standardization标准化,即提前全局BN
        data = np.reshape(data, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [height, width, bands])
        
        if 'Get Train,Val,Test ground truth':
            random.seed(curr_seed)
            gt_reshape = np.reshape(gt, [-1])
            train_rand_idx = []
            val_rand_idx = []
            if samples_type == 'ratio':
                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    rand_idx = random.sample(rand_list,
                                             np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class = idx[rand_idx]
                    train_rand_idx.append(rand_real_idx_per_class)
                train_rand_idx = np.array(train_rand_idx)
                train_data_index = []
                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index)
    
                ##将测试集（所有样本，包括训练样本）也转化为特定形式
                train_data_index = set(train_data_index)
                all_data_index = [i for i in range(len(gt_reshape))]
                all_data_index = set(all_data_index)
    
                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)
                test_data_index = all_data_index - train_data_index - background_idx
    
                # 从测试集中随机选取部分样本作为验证集
                val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
                val_data_index = random.sample(test_data_index, val_data_count)
                val_data_index = set(val_data_index)
                test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
    
                # 将训练集 验证集 测试集 整理
                test_data_index = list(test_data_index)
                train_data_index = list(train_data_index)
                val_data_index = list(val_data_index)


            if samples_type == 'same_num':
                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    real_train_samples_per_class = train_samples_per_class
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    if real_train_samples_per_class > samplesCount:
                        real_train_samples_per_class = samplesCount
                        # val_samples_per_class=0
                    rand_idx = random.sample(rand_list,
                                             real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                    train_rand_idx.append(rand_real_idx_per_class_train)
                    # if val_samples_per_class>0:
                    #     rand_real_idx_per_class_val = idx[rand_idx[-val_samples_per_class:]]
                    #     val_rand_idx.append(rand_real_idx_per_class_val)
                train_rand_idx = np.array(train_rand_idx)
                val_rand_idx = np.array(val_rand_idx)
                train_data_index = []
                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index)
    
                # val_data_index = []
                # for c in range(val_rand_idx.shape[0]):
                #     a = val_rand_idx[c]
                #     for j in range(a.shape[0]):
                #         val_data_index.append(a[j])
                # val_data_index = np.array(val_data_index)
    
                train_data_index = set(train_data_index)
                # val_data_index = set(val_data_index)
                all_data_index = [i for i in range(len(gt_reshape))]
                all_data_index = set(all_data_index)
    
                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)
                test_data_index = all_data_index - train_data_index - background_idx
    
                # 从测试集中随机选取部分样本作为验证集
                val_data_count = int(val_samples)  # 验证集数量
                val_data_index = random.sample(test_data_index, val_data_count)
                val_data_index = set(val_data_index)
    
                test_data_index = test_data_index - val_data_index
                # 将训练集 验证集 测试集 整理
                test_data_index = list(test_data_index)
                train_data_index = list(train_data_index)
                val_data_index = list(val_data_index)

            # 获取训练样本的标签图
            train_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(train_data_index)):
                train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
                pass

            # 获取测试样本的标签图
            test_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(test_data_index)):
                test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
                pass

            Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

            # 获取验证集样本的标签图
            val_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(val_data_index)):
                val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
                pass
            
            
            # 输出各集合数量
            print('train set:', len(train_data_index))
            print('val set:', len(val_data_index))
            print('test set:', len(test_data_index))
        # #保存训练集gt
        # Draw_Classification_Map( np.reshape(train_samples_gt,[height,width] ),'result\\'+dataset_name+'_train_gt')
        # #保存验证集gt
        # Draw_Classification_Map( np.reshape(val_samples_gt,[height,width] ), 'result\\'+dataset_name+'_val_gt')
        # #保存测试集gt
        # Draw_Classification_Map( np.reshape(Test_GT,[height,width] ),'result\\'+dataset_name+'_test_gt')
        # #保存全局gt图
        # Draw_Classification_Map( np.reshape(train_samples_gt+test_samples_gt+val_samples_gt,[height,width] ),'result\\'+dataset_name+'_gt')
        
        
        def SpiltHSI(data, gt, split_size, class_count):
            '''
            split HSI data with given slice_number
            :param data: 3D HSI data
            :param gt: 2D ground truth
            :param split_size: [height_slice,width_slice]
            :return: splited data and corresponding gt
            '''
            global EDGE
            e = EDGE  # 补边像素个数
            def GT_To_One_Hot(gt, class_count):
                '''
                Convet Gt to one-hot labels
                :param gt:
                :param class_count:
                :return:
                '''
                GT_One_Hot = []  # 转化为one-hot形式的标签
                for i in range(gt.shape[0]):
                    for j in range(gt.shape[1]):
                        temp = np.zeros(class_count, dtype=np.float32)
                        if gt[i, j] != 0:
                            temp[int(gt[i, j]) - 1] = 1
                        GT_One_Hot.append(temp)
                GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
                return GT_One_Hot
            
            split_height = split_size[0]
            split_width = split_size[1]
            m, n, d = data.shape
            gt = np.reshape(gt, [m, n])
            GT = GT_To_One_Hot(gt, class_count)
            
            # 将无法整除的块补0变为可整除
            if m % split_height != 0 or n % split_width != 0:
                data = np.pad(data, [[0, split_height - m % split_height], [0, split_width - n % split_width], [0, 0]],
                              mode='constant')
                GT = np.pad(GT, [[0, split_height - m % split_height], [0, split_width - n % split_width], [0, 0]],
                            mode='constant')
            m_height = int(data.shape[0] / split_height)
            m_width = int(data.shape[1] / split_width)
            
            pad_data = np.pad(data, [[e, e], [e, e], [0, 0]], mode="constant")
            final_data = []
            for i in range(split_height):
                for j in range(split_width):
                    temp = pad_data[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e, :]
                    final_data.append(temp)
            gt_split = np.split(GT, split_height, 0)
            final_gt = []
            for i in range(gt_split.__len__()):
                temp_gt = np.split(gt_split[i], split_width, 1)
                for j in range(temp_gt.__len__()):
                    tt = temp_gt[j]
                    tt = np.pad(tt, [[e, e], [e, e], [0, 0]], mode="constant")
                    final_gt.append(tt)
            
            final_data = np.array(final_data)
            final_gt = np.array(final_gt)
            return final_data, final_gt
        
        
        # 将训练数据、验证数据、测试数据 分割为若干小块,同时将GT转化为One-hot形式
        Train_Split_Data, Train_Split_GT = SpiltHSI(data, train_samples_gt, [split_height, split_width], class_count)
        Val_Split_Data, Val_Split_GT = SpiltHSI(data, val_samples_gt, [split_height, split_width], class_count)
        Test_Split_Data, Test_Split_GT = SpiltHSI(data, Test_GT, [split_height, split_width], class_count)
        
        _, patch_height, patch_width, bands = Train_Split_Data.shape  # 分割后的高光谱数据块的三个维度
        Train_Number_perClass = np.sum(Train_Split_GT, 0, keepdims=False)
        Train_Number_perClass = np.sum(Train_Number_perClass, 0, keepdims=False)
        Train_Number_perClass = np.sum(Train_Number_perClass, 0, keepdims=False)
        
        
        def Split_GT_Mask(Split_GT):
            '''
            Get Split-Mask for Split-GT
            :param Split_GT:
            :return:
            '''
            
            def Get_HSI_mask(GT):
                '''
                :param GT: 3D
                :return:
                '''
                m, n, class_count = GT.shape
                temp_ones = np.ones([class_count])
                GT_Mask = np.zeros([m, n, class_count], np.float32)
                for i in range(m):
                    for j in range(n):
                        if np.sum(GT[i, j]) != 0:
                            GT_Mask[i, j] = temp_ones
                return GT_Mask
            
            GT_Mask_Split = []
            for i in range(Split_GT.shape[0]):
                Mask_temp = Get_HSI_mask(Split_GT[i])
                GT_Mask_Split.append(Mask_temp)
            GT_Mask_Split = np.array(GT_Mask_Split)
            return GT_Mask_Split
        
        
        # 获取训练集对应的掩模
        Train_Split_Mask = Split_GT_Mask(Train_Split_GT)
        Val_Split_Mask = Split_GT_Mask(Val_Split_GT)
        Test_Split_Mask = Split_GT_Mask(Test_Split_GT)
        
        
        # 获取对应形状的权重w和偏差b，带默认初始化操作
        def get_weight_variable(shape, name=None):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
        
        
        def get_bias_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        
        
        # 卷积操作和池化操作
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        
        def onv2d_depthwise(x, W):
            return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        
        def LocalPadding(layer, kernel_size):
            input_shape = layer.get_shape().as_list()
            num_batch = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            num_channels = input_shape[3]
            a = tf.reshape(layer, [num_batch, height * width, num_channels])
            b = tf.expand_dims(a, -2)
            c = tf.tile(b, [1, 1, kernel_size[0] * kernel_size[1], 1])
            d = tf.reshape(c, [num_batch, height, width, kernel_size[0], kernel_size[1], num_channels])
            f = tf.transpose(d, [0, 1, 3, 2, 4, 5])
            g = tf.reshape(f, [num_batch, height * kernel_size[0], width * kernel_size[1], num_channels])
            return g
        
        
        def LocalPixels(input_feature, kernel_size, name='LocalPixels'):
            '''
            The padding Tensor with local kernel_size[0]*kernel_size[1 ]pixels
            :param input_feature: 4D tensor, [batch, height, width, Chanel]
            :param kernel_size: 1D tensor The spatial size of Pooling, [size, size]
            :param name: 'null'
            :return: 5D Tensor [batch, kernel_size[0]*height, kernel_size[1]*width, Chanel]
            '''
            
            def topkmaxpooling(layer, kernel_size):
                kernel_size = kernel_size
                num_points = kernel_size[0] * kernel_size[1]
                input_shape = layer.get_shape().as_list()
                # 定义input feature map的shape参数
                num_batch = input_shape[0]
                height = input_shape[1]
                width = input_shape[2]
                
                # 中心点坐标矩阵
                x_center = tf.reshape(tf.tile(tf.range(width), [height]), [height * width, -1])
                x_center = tf.tile(x_center, [1, num_points])
                x_center = tf.reshape(x_center, [height, width, num_points])
                x_center = tf.tile(tf.expand_dims(x_center, 0), [num_batch, 1, 1, 1])
                
                y_center = tf.tile(tf.range(height), [width])
                y_center = tf.transpose(tf.reshape(y_center, [width, height]))
                y_center = tf.reshape(y_center, [height * width, -1])
                y_center = tf.tile(y_center, [1, num_points])
                y_center = tf.reshape(y_center, [height, width, num_points])
                y_center = tf.tile(tf.expand_dims(y_center, 0), [num_batch, 1, 1, 1])
                
                x_center = tf.cast(x_center, "float32")
                y_center = tf.cast(y_center, "float32")
                
                # regular grid R矩阵
                x = tf.linspace(-(kernel_size[0] - 1) / 2, (kernel_size[0] - 1) / 2, kernel_size[0])
                y = tf.linspace(-(kernel_size[1] - 1) / 2, (kernel_size[1] - 1) / 2, kernel_size[1])
                x, y = tf.meshgrid(x, y)
                x_spread = tf.transpose(tf.reshape(x, (-1, 1)))
                y_spread = tf.transpose(tf.reshape(y, (-1, 1)))
                x_grid = tf.tile(x_spread, [1, height * width])
                x_grid = tf.reshape(x_grid, [height, width, num_points])
                y_grid = tf.tile(y_spread, [1, height * width])
                y_grid = tf.reshape(y_grid, [height, width, num_points])
                x_grid = tf.tile(tf.expand_dims(x_grid, 0), [num_batch, 1, 1, 1])
                y_grid = tf.tile(tf.expand_dims(y_grid, 0), [num_batch, 1, 1, 1])
                
                x = tf.add_n([x_center, x_grid])
                y = tf.add_n([y_center, y_grid])
                
                # 将N*H*W*num_points转换为N*3H*3W
                x_new = tf.reshape(x, [num_batch, height, width, kernel_size[0], kernel_size[1]])
                x_new = tf.squeeze(tf.split(x_new, height, 1))
                x_new = tf.squeeze(tf.split(x_new, kernel_size[0], 3))
                x_new = tf.reshape(tf.split(x_new, height, 1),
                                   [kernel_size[0] * height, num_batch, kernel_size[1] * width])
                x_new = tf.squeeze(tf.split(x_new, num_batch, 1))
                
                y_new = tf.reshape(y, [num_batch, height, width, kernel_size[0], kernel_size[1]])
                y_new = tf.squeeze(tf.split(y_new, height, 1))
                y_new = tf.squeeze(tf.split(y_new, kernel_size[0], 3))
                y_new = tf.reshape(tf.split(y_new, height, 1),
                                   [kernel_size[0] * height, num_batch, kernel_size[1] * width])
                y_new = tf.squeeze(tf.split(y_new, num_batch, 1))
                
                return x_new, y_new  # 每行对应x,y的二维坐标
            
            x, y = topkmaxpooling(input_feature, kernel_size)
            with tf.variable_scope(name + "/_bilinear_interpolate"):
                kernel_size = kernel_size
                num_points = kernel_size[0] * kernel_size[1]
                input_shape = tf.shape(input_feature)
                num_batch = input_shape[0]
                height = input_shape[1]
                width = input_shape[2]
                num_channels = input_shape[3]
                
                # 一维数据
                x = tf.reshape(x, [-1])
                y = tf.reshape(y, [-1])
                
                # 数据类型转换
                x = tf.cast(x, "float32")
                y = tf.cast(y, "float32")
                
                # 找到四个格点
                x0 = tf.cast(tf.floor(x), "int32")
                y0 = tf.cast(tf.floor(y), "int32")
                

                ratio0 = tf.cast(kernel_size[0] / 2, tf.int32)
                ratio1 = tf.cast(kernel_size[1] / 2, tf.int32)
                x0 = x0 + ratio0
                y0 = y0 + ratio1
                input_feature = tf.pad(input_feature, [[0, 0], [ratio0, ratio0], [ratio1, ratio1], [0, 0]])
                # 把input_feature和coordinate X和Y都转换为二维，方便从中抽取值
                input_feature_flat = tf.reshape(input_feature, tf.stack([-1, num_channels]))
                
                dimension_2 = width + ratio0 * 2
                dimension_1 = (width + ratio0 * 2) * (height + ratio1 * 2)
                base = tf.range(num_batch) * dimension_1
                repeat = tf.transpose(
                    tf.expand_dims(tf.ones(shape=(tf.stack([num_points * (width) * (height), ]))), 1), [1, 0])
                repeat = tf.cast(repeat, "int32")
                base = tf.matmul(tf.reshape(base, (-1, 1)), repeat)
                base = tf.reshape(base, [-1])
                base_y0 = base + y0 * dimension_2
                index_a = base_y0 + x0
                
                # 计算四个格点的value
                value_a = tf.gather(input_feature_flat, index_a)
                outputs = tf.reshape(value_a, [num_batch, kernel_size[0] * height, kernel_size[1] * width, num_channels])
                a = tf.reshape(outputs, [num_batch, height, kernel_size[0], width, kernel_size[1], num_channels])
                f = tf.transpose(a, [0, 1, 4, 3, 2, 5])
                f = tf.reshape(f, [num_batch, kernel_size[0] * height, kernel_size[1] * width, num_channels])
                return f
        
        
        def spatialsimilarity(input, kernel_size, sigma):
            localpixels = LocalPixels(input, kernel_size)
            localpadding = LocalPadding(input, kernel_size)
            diss = tf.square(localpadding - localpixels)
            diss = tf.reduce_mean(diss, axis=-1, keep_dims=True)
            diss = tf.exp(-diss / (sigma ** 2))
            return diss
        
        
        def model(iter, last_layers_chanels, spatial_block_chanels):
            global patch_height, patch_width, bands, class_count
            data, label, mask = iter.get_next()
            [m, n, d] = data.shape.as_list()[1:4]
            bn_training = tf.placeholder(tf.bool)
            
            # 定义BN正则化函数
            def BN(layer):
                h_temp = tf.layers.batch_normalization(layer, momentum=0.7, training=bn_training, trainable=True,
                                                       renorm=True)  # , training=False,trainable=True,renorm=True
                return h_temp
            
            def SSConv(input, output_dims, similarity, name=None):
                a = BN(input)

                b = tf.layers.conv2d(a, output_dims, 1,
                                     padding='same')  # ,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.03)
                b = tf.nn.sigmoid(b)

                w_conv1 = get_weight_variable([kernelsize, kernelsize, output_dims, 1], name)
                b_conv1 = get_bias_variable([output_dims])
                
                h_fc_localpixels = LocalPixels(b, (kernelsize, kernelsize))
                f = tf.multiply(h_fc_localpixels, similarity)  # 加权
                c = tf.nn.depthwise_conv2d(f, w_conv1, [1, kernelsize, kernelsize, 1], 'VALID') + b_conv1
                c = tf.nn.sigmoid(c)  # 128

                return c
            

            input_img, real_label, real_label_mask = data, label, mask
            
            spatialimg = tf.layers.conv2d(input_img, 3, 1, (1, 1), padding='same', activation='sigmoid')  # softsign
            sigma = tf.Variable([SIGMA], dtype=tf.float32, trainable=True, name='sigma')  # 可学习的超参数
            
            similarity = spatialsimilarity(spatialimg, [kernelsize, kernelsize], sigma)  # [1,145*5,145*5,1]
            
            # 第一层 由于提前进行BN化，因此第一层无需再BN
            h_conv1 = SSConv(input_img, last_layers_chanels, similarity, name='conv1')
            
            # 第二层
            h_conv2 = SSConv(h_conv1, spatial_block_chanels, similarity, name='conv2')
            h_conv2 = tf.concat([h_conv2, h_conv1], axis=-1)  # 拼接后的维度应该为203维
            
            # 第三层 spatial block
            h_conv3 = SSConv(h_conv2, spatial_block_chanels, similarity, name='conv3')
            h_conv3 = tf.concat([h_conv3, h_conv2], axis=-1)  # 拼接后的维度应该为448维
            
            # 第四层 spatial block
            h_conv4 = SSConv(h_conv3, spatial_block_chanels, similarity, name='conv4')
            h_conv4 = tf.concat([h_conv4, h_conv3], axis=-1)  # 拼接后的维度应该为480维
            
            # 第五层 spatial block
            h_conv5 = SSConv(h_conv4, spatial_block_chanels, similarity, name='conv5')
            h_conv5 = tf.concat([h_conv5, h_conv4], axis=-1)  # 拼接后的维度应该为496维
            

            h_conv_final = h_conv5
            h_fc = tf.layers.conv2d(h_conv_final, class_count, 1, padding='same', use_bias=True)
            
            h_fc_masked = tf.multiply(h_fc, real_label_mask)
            h_fc_masked = tf.reshape(h_fc_masked, [m * n, class_count])
            
            real_labels = tf.reshape(real_label, [m * n, class_count])
            # weighted loss function
            if "weighted":
                h_fc_pool = tf.nn.softmax(h_fc, -1)
                h_fc_pool_reshape = tf.reshape(h_fc_pool, [m * n, class_count])
                
                ##  动态权重
                we = -tf.multiply(real_labels, tf.log(h_fc_pool_reshape))  # 每一个点的损失
                we = tf.multiply(we, tf.reshape(real_label_mask, [m * n, class_count]))
                we2 = tf.constant(Train_Number_perClass + 1, dtype=tf.float32)  # 每类训练样本个数 加1是为了防止除0
                we2 = 1. / we2  # weight of each class
                we2 = tf.expand_dims(we2, 0)
                we2 = tf.tile(we2, [m * n, 1])
                we = tf.multiply(we, we2)
                cross_entropy = tf.reduce_sum(we)

            h_fc_masked = tf.reshape(h_fc_masked, [m, n, class_count])
            return h_fc, h_fc_masked, cross_entropy, input_img, real_label, real_label_mask, spatialimg, sigma, bn_training, similarity
        
        
        # 将数据扩展一维，以满足网络输入需求
        Train_Split_Data = np.expand_dims(Train_Split_Data, 1)
        Val_Split_Data = np.expand_dims(Val_Split_Data, 1)
        Test_Split_Data = np.expand_dims(Test_Split_Data, 1)
        
        Train_Split_GT = np.expand_dims(Train_Split_GT, 1)
        Val_Split_GT = np.expand_dims(Val_Split_GT, 1)
        Test_Split_GT = np.expand_dims(Test_Split_GT, 1)
        
        Train_Split_Mask = np.expand_dims(Train_Split_Mask, 1)
        Val_Split_Mask = np.expand_dims(Val_Split_Mask, 1)
        Test_Split_Mask = np.expand_dims(Test_Split_Mask, 1)
        
        # 全图，用于获取整图分割图 (无效)
        All_HSI_Data, All_HSI_GT = SpiltHSI(data, np.reshape(gt, [-1]), [1, 1], class_count)
        All_HSI_Mask = Split_GT_Mask(All_HSI_GT)
        
        All_HSI_Data = np.expand_dims(All_HSI_Data, 1)
        All_HSI_GT = np.expand_dims(All_HSI_GT, 1)
        All_HSI_Mask = np.expand_dims(All_HSI_Mask, 1)
        
        if 'Set Dataset Pipeline':
            TRAIN_DATASET = tf.data.Dataset.from_tensor_slices((Train_Split_Data, Train_Split_GT, Train_Split_Mask)).repeat(1)
            VAL_DATASET = tf.data.Dataset.from_tensor_slices((Val_Split_Data, Val_Split_GT, Val_Split_Mask)).repeat(1)
            TEST_DATASET = tf.data.Dataset.from_tensor_slices((Test_Split_Data, Test_Split_GT, Test_Split_Mask)).repeat(1)
            
            iter = tf.data.Iterator.from_structure(TRAIN_DATASET.output_types, TRAIN_DATASET.output_shapes)
            train_init_op = iter.make_initializer(TRAIN_DATASET)
            val_init_op = iter.make_initializer(VAL_DATASET)
            test_init_op = iter.make_initializer(TEST_DATASET)

        
        # 构建模型
        h_fc, h_fc_masked, cross_entropy, input_img, real_label, real_label_mask, spatialimg, sigma, bn_training_flag, similarity = model(
            iter, First_Chanels, After_Chanels)
        # 训练 使用标签样本光谱信息
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        opt1 = tf.train.AdamOptimizer(learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate_sigma)
        with tf.control_dependencies(update_ops):
            train_step = opt1.minimize(cross_entropy)
            # SIGMA 权重更新
            sigma_loss = opt2.minimize(cross_entropy, var_list=[sigma])
            train_step = tf.group(train_step, sigma_loss)
        
        # 计算预测正确的样本个数——计算图
        correct_prediction = tf.equal(tf.argmax(tf.reshape(h_fc_masked, [patch_height * patch_width, class_count]), 1),
                                      tf.argmax(tf.reshape(real_label, [patch_height * patch_width, class_count]), 1))
        correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) - \
                        tf.cast(patch_height * patch_width - tf.count_nonzero(tf.reduce_sum(real_label, -1)), tf.float32)
        # 计算每类样本的个数
        number_perClass = tf.reduce_sum(tf.reshape(real_label, [-1, class_count]), 0)  # [16]
        # 返回每个样本的预测标签，从0开始
        predicted_label_perSample = tf.argmax(tf.reshape(h_fc, [-1, class_count]), -1)
        # 返回每个样本的实际标签，从0开始
        read_label_perSample = tf.argmax(tf.reshape(real_label, [-1, class_count]), -1)
        
        
        def Get_Evaluation_Info(sess, option_name='train'):
            global h_fc, train_init_op, val_init_op, test_init_op, split_height, split_width, EDGE, bn_training_flag
            [patch_height, patch_width] = h_fc.shape.as_list()[1:3]
            patch_height -= EDGE * 2
            patch_width -= EDGE * 2
            trainging_flag = True
            if option_name == 'train':
                op = train_init_op
            if option_name == 'val':
                op = val_init_op
                trainging_flag = False
            if option_name == 'test':
                op = test_init_op
                trainging_flag = False
            
            _correct_count = 0
            _total_loss = 0
            _number_perClass = np.zeros([class_count])
            _predicted_label_perSample = []
            _read_label_perSample = []
            now_sigma = 0
            
            OutPut = []  # 网络输出
            SpatialMap = []
            sess.run(op)
            while True:
                try:
                    cc, n_p, plp, rlp, loss, now_sigma, outputPatch, myspatialimg = sess.run(
                        [correct_count, number_perClass, predicted_label_perSample, read_label_perSample, cross_entropy, sigma,
                         h_fc, spatialimg], feed_dict={bn_training_flag: trainging_flag})
                    _correct_count += cc
                    _number_perClass += n_p
                    _predicted_label_perSample += (list(plp))
                    _read_label_perSample += (list(rlp))
                    _total_loss += loss
                    
                    OutPut.append(outputPatch[0, :, :, :])  #####################
                    SpatialMap.append(myspatialimg[0, :, :, :])  #############
                
                except tf.errors.OutOfRangeError:
                    break
            correct_perClass = np.zeros([class_count])
            _predicted_label_perSample = np.reshape(_predicted_label_perSample, -1)
            _read_label_perSample = np.reshape(_read_label_perSample, -1)
            
            numb_of_all_samples = int(np.sum(_number_perClass))
            for x in range(numb_of_all_samples):
                if _read_label_perSample[x] == _predicted_label_perSample[x]:
                    correct_perClass[_read_label_perSample[x]] += 1
            acc_perClass = correct_perClass / _number_perClass
            AA = np.average(acc_perClass)
            OA = _correct_count / numb_of_all_samples
        
            # 对OutPut进行组装
            HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width, class_count], dtype=np.float32)
            SpatialMap_stack = np.zeros([split_height * patch_height, split_width * patch_width, SpatialMap[0].shape[-1]],
                                        dtype=np.float32)
            for i in range(split_height):
                for j in range(split_width):
                    if EDGE == 0:
                        HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                           i * split_width + j][
                                                                                                                       EDGE:,
                                                                                                                       EDGE:,
                                                                                                                       :]
                        SpatialMap_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = \
                            SpatialMap[i * split_width + j][EDGE:, EDGE:, :]
                    else:
                        HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                           i * split_width + j][
                                                                                                                       EDGE:-EDGE,
                                                                                                                       EDGE:-EDGE,
                                                                                                                       :]
                        SpatialMap_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = \
                        SpatialMap[i * split_width + j][EDGE:-EDGE, EDGE:-EDGE, :]
            HSI_stack = np.array(HSI_stack)
            SpatialMap_stack = np.array(SpatialMap_stack)
            
            return _total_loss, OA, AA, acc_perClass, now_sigma, HSI_stack, SpatialMap_stack
        
        def Get_similarity(sess, option_name='train'):
            global similarity
            [patch_height, patch_width] = h_fc.shape.as_list()[1:3]
            patch_height -= EDGE * 2
            patch_width -= EDGE * 2
            trainging_flag = True
            if option_name == 'train':
                op = train_init_op
            if option_name == 'val':
                op = val_init_op
                trainging_flag = False
            if option_name == 'test':
                op = test_init_op
                trainging_flag = False
            
            graph = tf.get_default_graph()
            conv1 = graph.get_tensor_by_name("conv1:0")
            conv2 = graph.get_tensor_by_name("conv2:0")
            conv3 = graph.get_tensor_by_name("conv3:0")
            conv4 = graph.get_tensor_by_name("conv4:0")
            conv5 = graph.get_tensor_by_name("conv5:0")
            
            similarity_list = []  # 相似度Map的输出
            sess.run(op)
            while True:
                try:
                    [mysimilarity, myconv1, myconv2, myconv3, myconv4, myconv5] = sess.run(
                        [similarity, conv1, conv2, conv3, conv4, conv5], feed_dict={bn_training_flag: trainging_flag})
                    similarity_list.append(mysimilarity[0, :, :, :])  #############
                except tf.errors.OutOfRangeError:
                    break
            
            # 对OutPut进行组装
            patch_height = patch_height * kernelsize
            patch_width = patch_width * kernelsize
            
            similarity_stack = np.zeros([split_height * patch_height, split_width * patch_width, 1], dtype=np.float32)
            for i in range(split_height):
                for j in range(split_width):
                    if EDGE == 0:
                        similarity_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = \
                        similarity_list[
                            i * split_width + j][
                        EDGE:,
                        EDGE:,
                        :]
                    
                    else:
                        similarity_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = \
                        similarity_list[
                            i * split_width + j][
                        EDGE * kernelsize:-EDGE * kernelsize,
                        EDGE * kernelsize:-EDGE * kernelsize,
                        :]
            
            similarity_stack = np.array(similarity_stack)
            # SpatialMap_stack = np.array(SpatialMap_stack)
            
            return similarity_stack, myconv1, myconv2, myconv3, myconv4, myconv5
        
        # 配置运行环境
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))  #
        sess.run(tf.global_variables_initializer())
        
        # 保存模型类
        saver = tf.train.Saver(var_list=tf.global_variables())
        # 记录模型训练中的最佳参数
        best_acc = 0
        best_loss = 1e9
        ff = open('result\\' + dataset_name + '_sigma.txt', 'a+')
        # 训练模型
        time_train_start = time.clock()
        for i in range(max_epoch + 1):
            # 训练一步
            sess.run(train_init_op)
            train_loss = 0
            while True:
                try:
                    _, loss = sess.run([train_step, cross_entropy], feed_dict={bn_training_flag: True})
                    train_loss += loss
                except tf.errors.OutOfRangeError:
                    break
            
            if i % 10 == 0:
                train_loss, train_OA, train_AA, train_acc_perClass, now_sigma, _, _ = Get_Evaluation_Info(sess, 'train')
                val_loss, val_OA, val_AA, val_acc_perClass, _, _, _ = Get_Evaluation_Info(sess, 'val')
                
                # output loss、train_acc、val_acc
                print('step:', i, 'train_OA=', train_OA, 'val_loss=', val_loss, 'val_OA=', val_OA, 'sigma=', now_sigma)
                
                # save sigma
                curr_sigma = str(i) + "\t" + str(now_sigma.tolist()[0]) + "\n"
                ff.write(curr_sigma)
                # save the better model
                if val_loss < best_loss:
                    best_acc = val_OA
                    best_loss = val_loss
                    saver.save(sess, "model\\best_model")
                    
        
        print('best val loss = {}'.format(best_loss))
        ff.close()
        
        time_train_end = time.clock()
        print('training complete.Training time=', time_train_end - time_train_start)
        
        # load model
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
            sess.as_default()
            saver.restore(sess, "model\\best_model")
        
            time_test_start = time.clock()
            test_loss, test_OA, test_AA, test_acc_perClass, _, output_data_background, SpatialMap = Get_Evaluation_Info(sess,
                                                                                                                        'test')
            time_test_end = time.clock()
            
            # # save the similarity map
            # similarityMap, myconv1, myconv2, myconv3, myconv4, myconv5 = Get_similarity(sess, 'test')
            # sio.savemat('result\\' + dataset_name + '_similarity_map.mat', {'data': similarityMap[:, :, 0]})
            # sio.savemat('result\\' + dataset_name + '_kernels.mat',
            #             {'conv1': myconv1, 'conv2': myconv2, 'conv3': myconv3, 'conv4': myconv4,
            #              'conv5': myconv5})
            
            # calculate AA
            output_data_background = output_data_background[0:height, 0:width, :]
            
            output_data = np.multiply(output_data_background, np.ceil(np.expand_dims(gt, -1) * 0.001))  # remove background
            output_data = np.reshape(output_data, [height * width, class_count])
            idx = np.argmax(output_data, axis=-1)
            zero_vector = np.zeros([class_count], dtype=np.float32)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            idx = idx + train_samples_gt
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(test_samples_gt)):
                if test_samples_gt[x] != 0:
                    count_perclass[int(test_samples_gt[x] - 1)] += 1
                    if test_samples_gt[x] == idx[x]:
                        correct_perclass[int(test_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)
            
            # calculate KPP
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [height * width, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [height, width])
            for ii in range(height):
                for jj in range(width):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16))
            test_kpp = kappa
            
            # 输出
            print("test OA=", test_OA, "AA=", test_AA, 'kpp=', test_kpp)
            print('acc per class:')
            print(test_AC_list)
            
            OA_ALL.append(test_OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)
            
            # save classification map with (without) background
            print('save picture...')
            all_pixel_pre_labels = output_data_background
            all_pixel_pre_labels = np.reshape(all_pixel_pre_labels, [height * width, class_count])
            idx = np.argmax(all_pixel_pre_labels, axis=-1)
            idx += 1
            idx = np.reshape(idx, [height, width])
            Draw_Classification_Map(idx, 'result\\' + dataset_name + '_with_background_' + str(train_ratio)+str(test_OA))
            sio.savemat('result\\' + dataset_name + '_classify_mat.mat', {'data': idx})
            
            idx = np.reshape(idx, [-1])
            idx[list(background_idx)] = 0
            idx = np.reshape(idx, [height, width])
            Draw_Classification_Map(idx, 'result\\' + dataset_name + '_without_background_' + str(train_ratio))
            
            # save classification results
            f = open('result\\' + dataset_name + '_results.txt', 'a+')
            str_results = '\n======================' \
                          + " learning rate=" + str(learning_rate) \
                          + " epochs=" + str(max_epoch) \
                          + " train ratio=" + str(train_ratio) \
                          + " val ratio=" + str(val_ratio) \
                          + " ======================" \
                          + "\nOA=" + str(test_OA) \
                          + "\nAA=" + str(test_AA) \
                          + '\nkpp=' + str(test_kpp) \
                          + '\ntrain time:' + str(time_train_end - time_train_start) \
                          + '\ntest time:' + str(time_test_end - time_test_start) \
                          + '\nacc per class:' + str(test_AC_list) + "\n"
            f.write(str_results)
            f.close()
            
            # SpatialMap (guide map)
            SpatialMap = np.reshape(SpatialMap[0:height, 0:width, :], [height * width, -1])
            minMax = preprocessing.MinMaxScaler()
            SpatialMap = minMax.fit_transform(SpatialMap)
            SpatialMap = np.reshape(SpatialMap, [height, width, -1])
            # plt.figure('Spatial similarity map')
            # plt.imsave('result\\' + dataset_name + '_SpatialMap.png', SpatialMap[:,:,0:3],dpi=300)
            Save_Image(SpatialMap[:, :, 0:3], 'result\\' + dataset_name + '_SpatialMap')
            sio.savemat('result\\' + dataset_name + '_SpatialMap.mat',
                        {'data': SpatialMap})
            # plt.show()
        tf.reset_default_graph()
        sess.close()
        
        print('complete.')
    
    
    
    OA_ALL=np.array(OA_ALL)
    AA_ALL=np.array(AA_ALL)
    KPP_ALL=np.array(KPP_ALL)
    AVG_ALL=np.array(AVG_ALL)
    
    print("\ntrain_ratio={}".format(curr_train_ratio),"\n==============================================================================")
    print('OA=',np.mean(OA_ALL),'+-',np.std(OA_ALL))
    print('AA=',np.mean(AA_ALL),'+-',np.std(AA_ALL))
    print('Kpp=',np.mean(KPP_ALL),'+-',np.std(KPP_ALL))
    print('AVG=',np.mean(AVG_ALL,0),'+-',np.std(AVG_ALL,0))
    


