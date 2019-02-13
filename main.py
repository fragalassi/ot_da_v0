from training_testing import train_isensee2017, predict, evaluate, create_test
import config

conf = config.Config(test=True)

train = train_isensee2017.Train_Isensee(conf)
train.main(overwrite=conf.overwrite)

test = create_test.Test(conf)
test.main(overwrite=conf.overwrite)

eval = evaluate.Evalaute(conf)
eval.main()

pred = predict.Predict(conf)
pred.main()



