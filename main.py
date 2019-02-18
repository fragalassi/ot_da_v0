from training_testing import train_isensee2017, predict, evaluate, create_test
import config
import sys
from Config import create_config

sys.path.append('/udd/aackaouy/OT-DA/')

df = create_config.create_conf(batch_size_l = [1, 8], initial_lr_l = [1e-4, 1e-7],
                loss_funcs = ["generalized_dice_loss", "weighted_dice_coefficient_loss"],
                depth_l = [3, 8], n_exp = 5)

for i in range(df.shape[0]):
    print("Testing config: ")
    print("=========")
    print(df.iloc[i])
    print("=========")
    conf = config.Config(test=False, rev=i, batch_size=df["Batch Size"].iloc[i],
                         initial_lr=df["Initial Learning Rate"].iloc[i],
                         loss_function=df["Loss function"].iloc[i],
                         depth=df["Depth"].iloc[i])

    train = train_isensee2017.Train_Isensee(conf)
    train.main(overwrite=conf.overwrite)

    test = create_test.Test(conf)
    test.main(overwrite=conf.overwrite)

    pred = predict.Predict(conf)
    pred.main()

    eval = evaluate.Evaluate(conf)
    eval.main()
