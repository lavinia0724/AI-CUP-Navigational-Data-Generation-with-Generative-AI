import torch


from model import UNext


from hyper_parameter import(device,
                            learning_rate,
                            max_epoch)



from get_data import get_data


from loss import loss_function



model = UNext(num_classes=3)
# model = torch.load("model1.pt")
model = model.train()

model = model.to(device=device)

# print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
# print(sum([p.numel() for p in model.parameters()]))



optimizer = torch.optim.Adam(model.parameters() , lr = learning_rate)
# optimizer = torch.optim.SGD(model.parameters() , lr = learning_rate)


for epoch in range(max_epoch):

    data , label = get_data()

    # data.shape  [batch_size , 3 , 240 , 428]
    # label.shape [batch_size , 3 , 240 , 428]

    predict = model(data)

    loss = loss_function(predict , label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        obsloss = ((predict - label) ** 2).mean()

    print(f"epoch = {epoch+1} -- loss = {loss.item():.4f} -- obsloss = {obsloss.item():.4f}")

    if (epoch + 1) % 500 == 0 :
        torch.save(model , "model1.pt")