from training_testing import train_isensee2017, predict, evaluate, create_test
import config
import sys

sys.path.append('/udd/aackaouy/OT-DA/')

conf = config.Config(test=True)

train = train_isensee2017.Train_Isensee(conf)
train.main(overwrite=conf.overwrite)

test = create_test.Test(conf)
test.main(overwrite=conf.overwrite)

pred = predict.Predict(conf)
pred.main()

eval = evaluate.Evaluate(conf)
eval.main()

