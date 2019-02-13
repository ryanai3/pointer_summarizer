from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import errno

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch import optim

from torch.nn import init

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def save_best_so_far(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        best_dir = os.path.join(self.model_dir, 'best')
        model_save_path = os.path.join(best_dir, 'model_%d_%d' % (iter, int(time.time())))
        try:
            os.makedirs(best_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        torch.save(state, model_save_path)

    def _change_lr(self, lr, it):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        lr_sum = tf.Summary()
        lr_sum.value.add(tag='lr', simple_value=lr)
        self.summary_writer.add_summary(lr_sum, it)

    def change_lr(self, it):
        it = it + 1
        new_lr = config.base * min(it ** -0.5, it * (config.warmup ** -1.5))
        self._change_lr(new_lr, it-1)

    def change_lr_lin(self, it):
        it_a = (config.anneal_steps - it) / config.anneal_steps
        lr_diff = config.start_lr - config.end_lr
        new_lr = config.end_lr  + lr_diff * max(0, it_a)
        self._change_lr(new_lr, it)

    def init_model(self):
        pass
#        for param in self.model.parameters():
#            init.uniform_(param, -0.2, 0.2)
#            init.normal_(param, 0, 0.15)
#            init.xavier_uniform


    def setup_optimizer(self):
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr



#        self.optimizer = optim.Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
#        self.optimizer = optim.RMSprop(params, momentum=0.9)
        self.optimizer = optim.Adam(params) #, betas=(0.9, 0.98)) #, betas = (-0.5, 0.999))
#        self.optimizer = optim.Adamax(params)
#        self.optimizer = optim.Adam(params, lr=0.001)
#        self.optimizer = optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)
        print(self.model)

        self.init_model()
        self.setup_optimizer()

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if (not (config.is_coverage or config.scratchpad)) or config.load_optimizer_override:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch, it):
#        self.change_lr(it)

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)


        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            if config.scratchpad:
                final_dist, s_t_1, _, attn_dist, p_gen, encoder_outputs = \
                    self.model.decoder(
                        y_t_1, s_t_1, encoder_outputs, encoder_feature, \
                        enc_padding_mask, c_t_1, extra_zeros, \
                        enc_batch_extend_vocab, coverage, di \
                    )
            else:
                final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = \
                    self.model.decoder(
                        y_t_1, s_t_1, encoder_outputs, encoder_feature, \
                        enc_padding_mask, c_t_1, extra_zeros, \
                        enc_batch_extend_vocab, coverage, di \
                    )

            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        if it % config.update_every == 0:
          self.optimizer.step()
          self.optimizer.zero_grad()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        best_loss = 20
        best_iter = 0
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch, iter)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

#            is_new_best = (running_avg_loss < best_loss) and (iter - best_iter >= 100)
#            best_loss = min(running_avg_loss, best_loss)

            if iter % 20 == 0:
                self.summary_writer.flush()
            print_interval = 100
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 2500 == 0:
                self.save_model(running_avg_loss, iter)
#            if is_new_best and iter > 200:
#                print('BEST steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
#                                                                           time.time() - start, best_loss))
#                self.save_best_so_far(running_avg_loss, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
