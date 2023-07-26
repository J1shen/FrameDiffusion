from model import Unet
import torch
import torch.nn.functional as F
from datagenerator import create_dataset
model = Unet(
            dim=256,
            channels=3,
            with_pose_emb=True
        )
train_loader,val_loader = create_dataset()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train(model,train_loader,val_loader,optimizer,epochs=10):
   device = "cuda" if torch.cuda.is_available() else "cpu"
   for epoch in range(epochs):
      for step, batch in enumerate(train_loader):
         optimizer.zero_grad()
        
         batch_size = batch["image_ori"].shape[0]
         batch = batch.to(device)

         predict = model(batch['image_ori'],p = batch['keys_trans']-batch['keys_ori'])
         loss = F.smooth_l1_loss(predict, batch['image_trans'])
            
         if step % 100 == 0:
            print("Loss:", loss.item())
            
            loss.backward()
            optimizer.step()

   return model

model = train(model,train_loader,val_loader,optimizer,epochs=10)