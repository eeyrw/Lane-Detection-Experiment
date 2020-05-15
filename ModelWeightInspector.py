import torch 

model_param = torch.load(r".\mobilenetv3_small_citys_best_model.pth",map_location=torch.device('cpu')) 

# restore_param = ['classifier.2.bias']
# 当然 如果你的目的是不想导入某些层的权重，将下述代码改为`if not k in restore_param`
# restore_param = {v for k, v in model.state_dict().items() if k in restore_param}
# print(restore_param)

for k, v in model_param.items():
    print('%s: %s'%(k,v.shape))