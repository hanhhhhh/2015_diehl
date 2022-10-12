import torch
def accuracy(model,dataloader):
    with torch.no_grad():
        model.eval()
        correct_num=0
        total_num=0
        for i,(batch_x,batch_y) in enumerate(dataloader):
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            output=model(batch_x)
            correct_num+=(output.argmax(dim=1)==batch_y).sum().item()
            total_num+=batch_x.shape[0]
    return correct_num/total_num