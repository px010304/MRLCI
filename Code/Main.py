import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model_sparse import Model
from DataHandler import DataHandler
from Utils.Utils import *
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import random
import wandb
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch 
import warnings
from sklearn.decomposition import PCA
#import umap
#import seaborn as sns
#from collections import defaultdict


# Function to set random seed for reproducibility
def set_seed(seed):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        self.plot_save_path = './sandian'  # 设置 t-SNE 图的保存路径
        if not os.path.exists(self.plot_save_path):
        	os.makedirs(self.plot_save_path)
    

        print('DRUG', args.drug, 'GENE', args.gene)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Acc']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    # Function to create a formatted print statement
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    # Function to perform external testing
    def external_test_run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        return reses['Acc']

    # Function to train and evaluate the model

    def run(self):
	    self.prepareModel()
	    log('Model Prepared')
	
	
	    if args.load_model is not None:
	        self.loadModel()
	        stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
	    else:
	        stloc = 0
	        log('Model Initialized')
	
	    for ep in range(stloc, args.epoch):
	        tstFlag = (ep % args.tstEpoch == 0)
	        reses = self.trainEpoch()
	        train_loss = reses
	        log(self.makePrint('Train', ep, reses, tstFlag))
	        if tstFlag:
	            reses = self.testEpoch(ep)  # 添加当前周期数
	            if reses['Acc'] > aucMax:
	                aucMax = reses['Acc']
	                bestEpoch = ep
	            test_r = reses
	            log(self.makePrint('Test', ep, reses, tstFlag))
	
	    reses = self.testEpoch(args.epoch)  # 添加最后一个周期数
	    log(self.makePrint('Test', args.epoch, reses, True))
	    
	    return aucMax

    def prepareModel(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)


    def trainEpoch(self):
    
        self.model.train()
        trnLoader = self.handler.trnLoader
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, tem in enumerate(trnLoader):
            drugs, genes, labels = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()

#            positive_indices, negative_indices = self.handler.get_positive_negative_indices(drugs, genes)           
#            ceLoss, sslLoss = self.model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate, positive_indices, negative_indices)
            
            ceLoss, sslLoss = self.model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
            sslLoss = sslLoss * args.ssl_reg

            regLoss = calcRegLoss(self.model) * args.reg
            loss = ceLoss + regLoss + sslLoss
            epLoss += loss.item()
            epPreLoss += ceLoss.item()
            self.opt.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.opt.step()
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    

    def testEpoch(self, epoch):

        self.model.eval()
        tstLoader = self.handler.tstLoader
        features_list = []
        labels_list = []
        predictions_list = []
        
        with torch.no_grad():
            for tem in tstLoader:
                drugs, genes, labels = tem
                drugs = drugs.long().cuda()
                genes = genes.long().cuda()
                labels = labels.long().cuda()
           
                pre = self.model.predict(self.handler.torchBiAdj, drugs, genes)
                features_list.append(pre.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
         
                predictions_list.append(pre.cpu().numpy())
        
        all_features = np.vstack(features_list)
        all_labels = np.concatenate(labels_list)
        all_predictions = np.vstack(predictions_list)
     
        self.plot_scatter(all_labels, all_predictions, epoch)
        
        # 调用 plot_tSNE，并传递保存路径
#        self.plot_tSNE(all_features, all_labels, epoch, self.plot_save_path)
        
        pre = F.log_softmax(pre, dim=1)
        pre = pre.data.max(1, keepdim=True)[1].detach().cpu()
        labels = labels.detach().cpu()
        epAcc = accuracy_score(labels, pre)
        
        ret = {'Acc': epAcc}
        return ret



    def plot_scatter(self, true_values, predicted_values, epoch): 
   
        true_values = np.array(true_values).flatten()
        predicted_values = np.array(predicted_values).flatten() 
          	
        min_length = min(len(true_values), len(predicted_values))
        true_values = true_values[:min_length]
        predicted_values = predicted_values[:min_length]  

        mask = (true_values == 1) 
        filtered_true_values = true_values[mask]
        filtered_predicted_values = predicted_values[mask]

        filtered_predicted_values = filtered_predicted_values[filtered_predicted_values >= 0.5]
        filtered_true_values = filtered_true_values[:len(filtered_predicted_values)] 

         
     
        plt.figure(figsize=(6, 6))
        plt.scatter(true_values, predicted_values, color="blue", s=10, alpha=0.5)
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2, label='1:1 line')

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
#        plt.title(f"Actual vs Predicted Values - Epoch {epoch}")



        plt.savefig(os.path.join(self.plot_save_path, f"scatter_epoch_{epoch}.png"))
        plt.close()
        
       

    # Function to load a pre-trained model
    def loadModel(self):
#        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))

        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'), strict=False)
        print("Current model state keys:", self.model.state_dict().keys())
        print("Loaded state keys:", t.load('../Models/' + args.load_model + '.pkl').keys())



        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

    # Function to save the trained model
    def save_model(self, model_path):
        model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        if not os.path.exists(model_parent_path):
            os.mkdir(model_parent_path)
        t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))


# Main execution block
if __name__ == '__main__':
    # if args.is_debug is True:
    #     print("DEBUGGING MODE - Start without wandb")
    #     wandb.init(mode="disabled")
    # else:
    #     wandb.init(project='HC', config=args)
    #     wandb.run.log_code(".")

    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    config = dict()
    results = list()
    best_result = None
    best_config = None

    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result = coach.run()
        results.append(result)

#    avg_r = np.mean(np.array(results), axis=0)

    
    avg_r = result
    std_r = np.std(results, axis=0)
    print('test results: ')
    print(avg_r)
    print(std_r)

    results.append(avg_r)
    results.append(std_r)

#    results_parent_path = os.path.join(wandb.run.dir, 'results')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_parent_path = os.path.join(current_dir, 'results')
    
    if not os.path.exists(results_parent_path):
        os.mkdir(results_parent_path)
    np.savetxt('{}/{}_result.txt'.format(results_parent_path, args.data), np.array(results), delimiter=",", fmt='%f')

    print('result saved!!!')
    # wandb.finish()
