import os

root_dir = os.path.join(
  os.path.expanduser("~"),
  "src/ryanai3/scratchpad/pointer_summarizer"
)

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "data/finished_files/chunked/train_*")
#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "data/finished_files/val.bin")
#eval_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "data/finished_files/test.bin")
#decode_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "data/finished_files/vocab")
#vocab_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/vocab")
log_root = os.path.join(root_dir, "logs/log")
#log_root = os.path.join(root_dir, "ptr_nw/log")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 16
update_every = 1
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

#start_lr = 0.1
#end_lr = 0.001
#anneal_steps = 50 * 1000

lr=0.15
#lr = 0.15 * 0.1
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=1.0

#base = 0.002
#base = 0.01
#warmup=2000

pointer_gen = True
is_coverage = False
cov_loss_wt = 0.0

pretrain = False
scratchpad = True

write_attn_fn = 'sigmoid'

load_optimizer_override = False

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15
