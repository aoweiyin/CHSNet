import matplotlib.pyplot as plt
import os

p_dict = {'Add_bbox_with_1': 0, 'Add_bbox_with_1_but_square': 1, 
          "Add_bbox_with_not_1": 2, "Add_both": 3, "Add_new_loss": 4}
p_dict_inv = {k:v for v,k in p_dict.items()}
color_dict = {0: "red", 1: "green", 2: "b", 3: "c", 4:"m"}
epoch = []
loss = []
mse = []
mae = []

for idx, p in enumerate(p_dict):
    p = os.path.join("../report/", p, "train.log")
    with open(p, 'r') as f:
        origins = f.readlines()
    # flog = open("test.out", "w")
    epochi = []
    lossi = []
    msei = []
    maei = []
    i = 0
    while i<len(origins):
        if origins[i][15] == '-':
            i += 1
            if i>=len(origins):
                break
            anno = origins[i][15:]
            if anno[0]!='E':
                i+=1
                continue
            sl = anno.split(" ")
            sl = [si.strip(",") for si in sl]
            if int(sl[1])>100:
                break
            epochi.append(int(sl[1]))
            lossi.append(float(sl[4]))
            msei.append(float(sl[6]))
            maei.append(float(sl[8]))
            # flog.write(anno)
        else :
            i+=1
            continue
    # flog.write(origins[i])
    epoch.append(epochi)
    loss.append(lossi)
    mse.append(msei)
    mae.append(maei)

for idx, _mse in enumerate(mse):
    plt.plot(epoch[0], _mse, 
                    color=color_dict[idx], label=p_dict_inv[idx]+"-mse")
    plt.plot(epoch[0], mae[idx], 
                    color=color_dict[idx], label=p_dict_inv[idx]+"-mae")
plt.title("mse&mae with epoch")
plt.legend()
plt.show()
    


# ex = "05-27 17:58:37 Epoch 106 Train, Loss: 260935.01, MSE: 2146.73 MAE: 447.15, Cost 124.0 sec"

# s = ex[15:]
# sl = s.split(" ")
# sl = [si.strip(",") for si in sl]
# epoch = []
# loss = []
# mse = []
# mae = []
# epoch.append(int(sl[1]))
# loss.append(float(sl[4]))
# mse.append(float(sl[6]))
# mae.append(float(sl[8]))

# print(epoch)
# print(loss)
# print(mse)
# print(mae)