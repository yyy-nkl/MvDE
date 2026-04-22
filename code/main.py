# import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')

# parser.add_argument('-lgbn_estimators', type=int, help='lgbn_estimators', default=40)
# parser.add_argument('-CNNn_estimators', type=int, help='CNNn_estimators', default=40)
# parser.add_argument('-lgblearning_rate', type=float, help='lgblearning_rate', default=0.01)
# parser.add_argument('-CNNlearning_rate', type=float, help='CNNlearning_rate', default=0.01)
# parser.add_argument('-CNNepochs', type=int, help='CNNepochs', default=100)
# parser.add_argument('-batch_size', type=int, help='batch_size', default=100)
# parser.add_argument('-alpha', type=float, help='alpha', default=0.1)
# parser.add_argument('-out_dirname',type=str, help='out_dirname', required=True)

# myargs = parser.parse_args()

#========================================================================================

import itertools
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
from tensorflow.python.keras.engine.training_utils import prepare_sample_weight_modes
from classifiers import *
from GAE_trainer import *
from GAE import *
from NMF import *
from metric import *
from similarity_fusion import *
from five_AE import *
import warnings
import os
import time

import Code.test2_CNN as test2_CNN
from Code.multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
import lightgbm as lgb
from scipy.interpolate import interp1d
#========================================================================================




os.chdir('/home/user/guoyu/mt/1.othermethods/2.MDA-AENMF-main/MDA-AENMF-main/')

for out_dirname in range(500):

    path = f'./new/find_best/{out_dirname}/'
    if not os.path.exists(path):
        os.makedirs(path)

    start = time.perf_counter()  # 记录开始时间
    warnings.filterwarnings("ignore")
    # extract feature parameter
    n_splits = 5
    classifier_epochs = 50
    m_threshold = [0.7]
    epochs=[200]
    fold = 0
    #initialization
    result = np.zeros((1, 7), float)
    pre_matrix = np.zeros((1, 5), float)
    #adamcnn parameter
    lgbn_estimators = 40
    CNNn_estimators = 40
    CNNlearning_rate = lgblearning_rate = 0.01
    CNNepochs = 100
    batch_size = 100
    n_features = 308
    alpha=0.1
    #out_dirname = 'bs500'
    tprs=[]
    aucs=[]
    pres=[]
    recs=[]
    mean_fpr=np.linspace(0,1,100)
    mean_recall = np.linspace(0, 1, 100)

    for s in itertools.product(m_threshold,epochs):

            association = pd.read_csv("./M_D.csv", index_col=0).to_numpy()  #(2262, 216)
            samples = get_all_samples(association) #(9072, 3)；一半pos,一半neg


            k1 = 226
            k2 = 21
            # m_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)  # Integration of similarity networks for metabolites or diseases
            m_fusion_sim = pd.read_csv("./m_fusion_sim.csv", index_col=0, dtype=np.float32).to_numpy()
            d_fusion_sim = pd.read_csv("./d_fusion_sim.csv", index_col=0, dtype=np.float32).to_numpy()

            kf = KFold(n_splits=n_splits, shuffle=True)

            # Metabolite and disease features extraction from NMF
            D = 90
            NMF_mfeature, NMF_dfeature = get_low_feature(D, 0.01, pow(10, -4), association)  #(2262, 90); (216, 90)就是文件feature_MFd.csv和feature_MFm.csv
            #np.save('./NMF_mfeature.npy', NMF_mfeature)
            #np.save('./NMF_dfeature.npy', NMF_dfeature)

            #NMF_mfeature = np.load('./NMF_mfeature.npy')
            #NMF_dfeature = np.load('./NMF_dfeature.npy')

            with open(path +'/zhibiao.txt', 'a') as zhibiao:
                zhibiao.write('parameters setting:' + '\n')
                zhibiao.write('lgbn_estimators: ' + str(lgbn_estimators) + ', CNNn_estimators: ' +str(CNNn_estimators)+ ', lgblearning_rate: ' +str(lgblearning_rate) + ', CNNlearning_rate: ' +str(CNNlearning_rate) +', CNNepochs: '+str(CNNepochs) +  ', batch_size: ' +str(batch_size) + ', alpha: ' +str(alpha) +'\n')
                zhibiao.write('aupr, auc, f1_score, accuracy, recall, specificity, precision' + '\n')
            for train_index, val_index in kf.split(samples):
                fold += 1
                train_samples = samples[train_index, :]
                val_samples = samples[val_index, :]
                new_association = association.copy()
                for i in val_samples:
                    new_association[i[0], i[1]] = 0  #验证集的值都让为0;(2262, 216)

                # Metabolite features extraction from GAE
                m_network = sim_thresholding(m_fusion_sim, s[0])  #similarity_fusion.py中的函数；m_fusion这个矩阵中的值大于s[0]的，赋值为1，否则赋值为0
                m_adj, meta_features = generate_adj_and_feature(m_network, new_association) #GAE_trainer.py中的函数,生成图的邻接矩阵和特征矩阵,(2262, 2262);(2262, 216)
                m_features = get_gae_feature(m_adj, meta_features,s[1], 1)  #(2262, 64)

                # Disease features extraction from five-layer auto-encoder
                d_features = five_AE(d_fusion_sim)

                # get feature and label
                train_feature, train_label = generate_f1(D, train_samples, m_features, d_features, NMF_mfeature, NMF_dfeature)
                val_feature, val_label = generate_f1(D, val_samples, m_features, d_features, NMF_mfeature, NMF_dfeature)

                '''
                # MLP classfier
                model = BuildModel(train_feature, train_label)
                test_N = val_samples.shape[0]
                y_score = np.zeros(test_N)
                y_score = model.predict(val_feature)[:, 0]
                '''


                #model = lgb.LGBMClassifier(learning_rate=lgblearning_rate, n_estimators=lgbn_estimators)
                model = lgb.LGBMClassifier(learning_rate=lgblearning_rate, n_estimators=lgbn_estimators, eval_metric='logloss')


                X_tr = train_feature
                y_tr = train_label
                X_te = val_feature

                model.fit(X_tr, y_tr)
                lscore = model.predict_proba(X_te)
                lscore = lscore[:, 1]


                X_train_r = test2_CNN.reshape_for_CNN(X_tr)
                X_test_r = test2_CNN.reshape_for_CNN(X_te)
                bdt_real_test_CNN = Ada_CNN(base_estimator=test2_CNN.baseline_model(n_features=n_features),
                                            n_estimators=CNNn_estimators,
                                            learning_rate=CNNlearning_rate, epochs=CNNepochs)
                bdt_real_test_CNN.fit(X_train_r, y_tr, batch_size = batch_size)
                # cpre_label = bdt_real_test_CNN.predict(X_test_r)
                cscore = bdt_real_test_CNN.predict_proba(X_test_r)
                cscore = cscore[:, 1]
                score = alpha * cscore + (1-alpha) * lscore
                
                #allscore = np.column_stack(val_samples, )
                # calculate metrics
                fpr, tpr, thresholds = roc_curve(val_label, score)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                pre, rec_, _ = precision_recall_curve(val_label, score)
                # pres.append(interp(mean_rec, pre, rec_))
                # pres[-1][0] = 0.0

                interp_func = interp1d(rec_, pre, bounds_error=False, fill_value="extrapolate")

                # 使用插值函数计算与mean_recall对齐的Precision值
                interp_pre = interp_func(mean_recall)

                # 将插值后的Precision和Recall值添加到列表中
                pres.append(interp_pre)
                recs.append(mean_recall)

                evl, pre_val_label = get_metrics(val_label, score)
                pre_matrix_temp = np.column_stack((val_samples, pre_val_label, score))
                pre_matrix=np.row_stack((pre_matrix, pre_matrix_temp))

                result += evl
                print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]',
                    evl)

                with open(path +'/zhibiao.txt', 'a') as zhibiao:
                    zhibiao.write(str(evl) + '\n')
            
                fpr_tpr =  np.column_stack((fpr, tpr))
                np.savetxt(f'{path}/fpr_tpr_fold{fold}.txt', fpr_tpr, delimiter=',', header='FPR,TPR', comments='')

                pre_rec =  np.column_stack((pre, rec_))
                np.savetxt(f'{path}/pre_rec_fold{fold}.txt', pre_rec, delimiter=',', header='PRE,REC', comments='')

            print("==================================================")
            avg = result / n_splits
            
            with open(path +'/zhibiao.txt', 'a') as zhibiao:
                zhibiao.write('average: ' + str(avg) + '\n')
            
            
            np.savetxt(f'{path}/pre_val_matrix.txt', pre_matrix, delimiter=',', header='mt, disease, label, pre_label, pre_score', comments='')


            #============plot auROC curve
            plt.figure(figsize=(8, 8))
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g', alpha=.8)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color='r', label=r'iMDA-adaCNN (area=%0.3f)' % mean_auc, lw=2, alpha=.8)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
            plt.xlim([0, 1.0])
            plt.ylim([0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.show()
            plt.savefig(f'{path}/AUC_curve.jpg', dpi=300, bbox_inches='tight', format='jpg')
            

            #+===========pr curve
            # 计算平均Precision和平均Recall
            mean_pre = np.mean(pres, axis=0)
            mean_rec = np.mean(recs, axis=0)
            mean_aupr = auc(mean_rec, mean_pre)

            # 绘制PR曲线
            plt.figure(figsize=(8, 8))
            plt.plot(mean_rec, mean_pre, color='r', label=r'iMDA-adaCNN (area=%0.3f)' % mean_aupr, lw=2, alpha=.8)

            # 绘制随机猜测的对角线
            plt.plot([0, 1], [1, 0], 'k--')

            # 设置图例、坐标轴标签和标题
            plt.xlim([0, 1.0])
            plt.ylim([0, 1.0])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="lower left")

            # 显示图表
            plt.show()
            # 保存图表为JPG文件
            plt.savefig(f'{path}/PR_curve.jpg', dpi=300, bbox_inches='tight', format='jpg')

end = time.perf_counter()  # 记录结束时间
run_time = (end - start)/3600  # 计算运行时间
print("运行时间：", run_time, "秒")
with open(path +'/zhibiao.txt', 'a') as zhibiao:
    zhibiao.write('using time: ' + str(run_time) + 'h' +'\n')
        
