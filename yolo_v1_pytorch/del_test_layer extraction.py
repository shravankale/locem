import torch
from r50_yolo import resnet50
from loss import Loss

S=7
B=2
X=5
C=30
image_size = 448

resume = '/mnt/data1/shravank/results/locem/main/run/g4_yolob_e300b64_v1/model_latest.pth'

model = resnet50(pretrained=True,S=S,B=B,C=C,X=X)
model = torch.nn.DataParallel(model).cuda()

criterion = Loss(feature_size=S, num_bboxes=B, num_classes=C, lambda_coord=5.0, lambda_noobj=0.5)

optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=1e-4)

loc = 'cuda:{}'.format(0)
#checkpoint = torch.load(resume, map_location=loc)
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)
print(model)

features = list(model.module.children())[:8]
f1 = nn.ModuleList(features).eval()

class Yolob(torch.nn.Module):
    def __init__(self):
        super(Yolob,self).__init__()

        features = list(model.module.children())[]
        self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        emb,out = [],[]
        for i,model in enumerate(self.features):
            x = model(x)
            if i==8:
                emb =x
            elif i ==10:
                out = x
            else:
                continue
        
        return emb,out

new_model = Yolob(model)
print('here')