from data_preprocessor import *
from AutoRec import AutoRec
import tensorflow as tf
import time
import argparse
current_time = time.time()

parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int,default=100)

parser.add_argument('--optimizer_method', choices=['Adam','RMSProp'],default='Adam')
parser.add_argument('--grad_clip', type=bool,default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50,help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

data_name = 'ml-1m'; num_users = 6040; num_items = 3952; num_total_ratings = 1000209; train_ratio = 0.9
path = "./data/%s" % data_name + "/"

result_path = './results/' + data_name + '/' + str(args.random_seed) + '_' + str(args.optimizer_method) + '_' + str(args.base_lr) + "_" + str(current_time)+"/"
R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(path, num_users, num_items,num_total_ratings, 1, 0, train_ratio)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    AutoRec = AutoRec(sess,args,
                      num_users,num_items,
                      R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                      user_train_set, item_train_set, user_test_set, item_test_set,
                      result_path)
    AutoRec.run()