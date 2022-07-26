from train import *

#----------------------------------------------------------------#
#----------------------------------------------------------------#

graphs_dir = "./data/graphs/graph.pkl"
graphs, adjs = load_graphs(graphs_dir ) # 365张图和邻接矩阵，注意点索引是1-782
label_dir = './data/2018_Leakages.csv'
df_label = load_label(label_dir) # 2018 leakage pipes dataset; 105120(365x288) rows × 14(leakages) columns

feats = []
for i in range(len(graphs)):
    feats.append(graphs[i].graph['feature'])

device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

dataset = MyDataset(args, graphs, feats, adjs, df_label)

dataloader = DataLoader(dataset,  # 定义dataloader # batch_size是512>=365,所以会导入2018年所有图的信息
                        batch_size=args.batch_size, # default 512
                        shuffle=True,
                        num_workers=10, 
                        collate_fn=MyDataset.collate_fn, 
                        drop_last=False # 是否扔掉len % batch_size
                        )

# 导入模型结构
model = DySAT(args, feats[0].shape[1], args.time_steps).to(device) 

# 导入模型参数
model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
model.eval()
emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                    train_edges_neg,
                                                    val_edges_pos, 
                                                    val_edges_neg, 
                                                    test_edges_pos,
                                                    test_edges_neg, 
                                                    emb, 
                                                    emb)
auc_val = val_results["HAD"][1]
auc_test = test_results["HAD"][1]
print("Best Test AUC = {:.3f}".format(auc_test))