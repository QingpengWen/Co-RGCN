# -*- coding: utf-8 -*-#
"""
@CreateTime :       2023/2/28 21:30
@Author     :       Qingpeng Wen
@File       :       process.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2023/3/10 23:25
"""

import os
import time
import math
import random

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import Counter

# Utils functions copied from Slot-gated model, origin url:
# 	https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
from transformers import get_linear_schedule_with_warmup
from utils import miulab
import fitlog

def multilabel2one_hot(labels, nums):
    res = [0.] * nums
    if len(labels) == 0:
        return res
    if isinstance(labels[0], list):
        for label in labels[0]:
            res[label] = 1.
        return res
    for label in labels:
        res[label] = 1.
    return res

def instance2onehot(func, num_intent, data):
    res = []
    for intents in func(data):
        res.append(multilabel2one_hot(intents, num_intent))
    return np.array(res)

def normalize_adj(mx):
    """
    Row-normalize matrix  D^{-1}A
    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx

class Processor(object):

    def __init__(self, dataset, model, optimizer, start_epoch, batch_size, args):
        self.__dataset = dataset
        self.__model = model
        self.__args = args
        self.__batch_size = batch_size
        self.__load_dir = args.load_dir
        self.start_epoch = start_epoch

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()
        self.__criterion_intent = nn.BCEWithLogitsLoss()

        t_total = math.ceil(self.__dataset.num_training_samples / self.__batch_size) * self.__dataset.num_epoch
        self.__optimizer = optimizer
        self.__scheduler = get_linear_schedule_with_warmup(self.__optimizer, num_warmup_steps=5,
                                                           num_training_steps=t_total)

        if self.__load_dir:
            if torch.cuda.is_available():
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'))
            else:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'),
                                          map_location=torch.device('cpu'))

    def train(self):
        best_dev_sent = 0.0
        best_test_sent = 0.0
        best_epoch = 0

        no_improve = 0
        dataloader = self.__dataset.batch_delivery('train')

        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0
            time_start = time.time()
            self.__model.train()

            for text_batch, slot_batch, intent_batch, raw_text in tqdm(dataloader, ncols=50):
                padded_text, [sorted_slot, sorted_intent], seq_lens = self.__dataset.add_padding(text_batch,
                                                                                                 [(slot_batch, True),
                                                                                                  (intent_batch, False)])

                sorted_intent_exp = []
                for item, num in zip(sorted_intent, seq_lens):
                    sorted_intent_exp.extend([item] * num)
                sorted_intent = [multilabel2one_hot(intents, len(self.__dataset.intent_alphabet)) for intents in
                                 sorted_intent_exp]

                text_var = torch.LongTensor(padded_text)
                slot_var = torch.LongTensor(sorted_slot)
                intent_var = torch.Tensor(sorted_intent)
                max_len = np.max(seq_lens)

                if torch.cuda.is_available():
                    text_var = text_var.cuda()
                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                random_slot, random_intent = random.random(), random.random()

                slot_out, intent_out = self.__model(text_var, raw_text, seq_lens)

                slot_var = torch.cat([slot_var[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0)
                slot_loss = self.__criterion(slot_out, slot_var)

                intent_out = torch.cat([intent_out[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0)
                intent_loss = self.__criterion_intent(intent_out, intent_var)
                intent_loss_alpha = self.__args.intent_loss_alpha
                slot_loss_alpha = self.__args.slot_loss_alpha
                batch_loss = slot_loss_alpha * slot_loss + intent_loss_alpha * intent_loss
                self.__optimizer.zero_grad()
                batch_loss.backward()
                self.__optimizer.step()
                self.__scheduler.step()

                try:
                    total_slot_loss += slot_loss_alpha * slot_loss.cpu().item()
                    total_intent_loss += intent_loss_alpha * intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss_alpha * slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss_alpha * intent_loss.cpu().data.numpy()[0]

            fflag = True
            cntt = 0
            while fflag:
                cntt += 1
                if cntt >= 100:
                    raise FileExistsError("gg")
                try:
                    time.sleep(1)
                    fitlog.add_loss(total_slot_loss, name='slot loss', step=epoch)
                    fflag = False
                except FileExistsError as e:
                    pass

            fitlog.add_loss(total_intent_loss, name='intent loss', step=epoch)
            fitlog.add_loss(total_intent_loss + total_slot_loss, name='total loss', step=epoch)
            time_con = time.time() - time_start
            print(
                '[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent data is {:2.6f}, cost ' \
                'about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))

            change, time_start = False, time.time()
            dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score = self.estimate(
                if_dev=True,
                test_batch=self.__batch_size,
                args=self.__args)
            test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = self.estimate(
                if_dev=False, test_batch=self.__batch_size, args=self.__args)
            print('Validation result: epoch: {}, slot f1 score: {:2.6f}, '
                  'intent f1 score: {:2.6f}, intent acc score: {:2.6f}, semantic acc: {:.2f}.'.format(
                   epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score))
            print('Test result: epoch: {}, slot f1 score: {:.6f}, intent f1 score: {:.6f}, intent acc score:'
                  ' {:.6f}, semantic accuracy score: {:.6f}.'.
                  format(epoch, test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc))
            fitlog.add_metric(
                {"dev": {"slot f1": dev_slot_f1_score,
                         "intent f1": dev_intent_f1_score,
                         "intent acc": dev_intent_acc_score,
                         "exact acc": dev_sent_acc_score
                         }
                 },
                step=epoch
            )

            if dev_sent_acc_score >= best_dev_sent:
                no_improve = 0
                best_epoch = epoch
                best_dev_sent = dev_sent_acc_score

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    try:
                        os.mkdir(model_save_dir)
                    except:
                        pass

                fitlog.add_best_metric(
                    {"dev": {"slot f1": dev_slot_f1_score,
                             "intent f1": dev_intent_f1_score,
                             "intent acc": dev_intent_acc_score,
                             "exact acc": dev_sent_acc_score
                             }
                     }
                )
                fitlog.add_metric(
                    {"test": {"slot f1": test_slot_f1,
                              "intent f1": test_intent_f1,
                              "intent acc": test_intent_acc,
                              "exact acc": test_sent_acc
                              }
                     },
                    step=epoch
                )
                fitlog.add_best_metric(
                    {"test": {"slot f1": test_slot_f1,
                              "intent f1": test_intent_f1,
                              "intent acc": test_intent_acc,
                              "exact acc": test_sent_acc
                              }
                     }
                )
                torch.save(self.__model, os.path.join(model_save_dir, "dev_model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dev_dataset.pkl'))

                print('\n[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, '
                      'the intent f1 score is {:2.6f}, the intent acc score is {:2.6f}, the semantic acc is {:2.6f}'.
                      format(epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score,
                      dev_sent_acc_score))

            if test_sent_acc >= best_test_sent:
                no_improve = 0
                best_epoch = epoch
                best_test_sent = test_sent_acc

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    try:
                        os.mkdir(model_save_dir)
                    except:
                        pass

                fitlog.add_best_metric(
                    {"dev": {"slot f1": dev_slot_f1_score,
                             "intent f1": dev_intent_f1_score,
                             "intent acc": dev_intent_acc_score,
                             "exact acc": dev_sent_acc_score
                             }
                     }
                )
                fitlog.add_metric(
                    {"test": {"slot f1": test_slot_f1,
                              "intent f1": test_intent_f1,
                              "intent acc": test_intent_acc,
                              "exact acc": test_sent_acc
                              }
                     },
                    step=epoch
                )
                fitlog.add_best_metric(
                    {"test": {"slot f1": test_slot_f1,
                              "intent f1": test_intent_f1,
                              "intent acc": test_intent_acc,
                              "exact acc": test_sent_acc
                              }
                     }
                )
                state = {'model': self.__model, 'dataset': self.__dataset, 'optimizer': self.__optimizer,
                         'epoch': epoch}
                torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))
                torch.save(state, os.path.join(model_save_dir, "model_epoch.pkl"))
                print('Test result: epoch: {}, Slot f1 score: {:.6f}, Intent f1 score: {:.6f}, Intent Acc:'
                      ' {:.6f}, Semantic Acc: {:.6f}.'.
                      format(epoch, test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc))

            else:
                no_improve += 1

            if self.__args.early_stop == True:
                if no_improve > self.args.patience:
                    print('early stop at epoch {}'.format(epoch))
                    break
        print('Best epoch is {}'.format(best_epoch))
        return best_epoch

    def estimate(self, if_dev, args, test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if if_dev:
            ss, pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "dev", test_batch, args)
        else:
            ss, pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "test", test_batch, args)

        num_intent = len(self.__dataset.intent_alphabet)
        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_f1_score = f1_score(
            instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, real_intent),
            instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, pred_intent),
            average='macro')
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        # Write those sample both have intent and slot errors.
        with open(os.path.join(args.save_dir, 'error.txt'), 'w', encoding="utf8") as fw:
            for p_slot_list, r_slot_list, p_intent_list, r_intent in \
                    zip(pred_slot, real_slot, pred_intent, real_intent):
                fw.write(','.join(p_intent_list) + '\t' + ','.join(r_intent) + '\n')
                for w, r_slot, in zip(p_slot_list, r_slot_list):
                    fw.write(w + '\t' + r_slot + '\t''\n')
                fw.write('\n\n')

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def validate(model_path, dataset, batch_size, num_intent, args):
        """
        validation will write mistaken samples to files and make scores.
        """

        if torch.cuda.is_available():
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))

        ss, pred_slot, real_slot, pred_intent, real_intent = Processor.prediction(
            model, dataset, "test", batch_size, args)

        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_f1_score = f1_score(instance2onehot(dataset.intent_alphabet.get_index, num_intent, real_intent),
                                   instance2onehot(dataset.intent_alphabet.get_index, num_intent, pred_intent),
                                   average='macro')
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        # Write those sample both have intent and slot errors.
        with open(os.path.join(args.save_dir, 'error.txt'), 'w', encoding="utf8") as fw:
            for p_slot_list, r_slot_list, p_intent_list, r_intent in \
                    zip(pred_slot, real_slot, pred_intent, real_intent):
                fw.write(','.join(p_intent_list) + '\t' + ','.join(r_intent) + '\n')
                for w, r_slot, in zip(p_slot_list, r_slot_list):
                    fw.write(w + '\t' + r_slot + '\t''\n')
                fw.write('\n\n')

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size, args):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        all_token = []
        for text_batch, slot_batch, intent_batch, raw_text in tqdm(dataloader, ncols=50):
            padded_text, [sorted_slot, sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(slot_batch, False), (intent_batch, False)],
                digital=False
            )
            real_slot.extend(sorted_slot)
            all_token.extend([pt[:seq_lens[idx]] for idx, pt in enumerate(padded_text)])
            for intents in list(Evaluator.expand_list(sorted_intent)):
                if '#' in intents:
                    real_intent.append(intents.split('#'))
                else:
                    real_intent.append([intents])

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = torch.LongTensor(digit_text)
            max_len = np.max(seq_lens)
            if torch.cuda.is_available():
                var_text = var_text.cuda()
            slot_idx, intent_idx = model(var_text, raw_text, seq_lens, n_predicts=1)
            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
            pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))
            intent_idx_ = [[] for i in range(len(digit_text))]
            for item in intent_idx:
                intent_idx_[item[0]].append(item[1])
            intent_idx = intent_idx_
            pred_intent.extend(dataset.intent_alphabet.get_instance(intent_idx))
        if 'MixSNIPS' in args.data_dir or 'MixATIS' in args.data_dir or 'DSTC' in args.data_dir:
            [p_intent.sort() for p_intent in pred_intent]
        with open(os.path.join(args.save_dir, 'token.txt'), "w", encoding="utf8") as writer:
            idx = 0
            for line, slots, rss in zip(all_token, pred_slot, real_slot):
                for c, sl, rsl in zip(line, slots, rss):
                    writer.writelines(
                        str(sl == rsl) + " " + c + " " + sl + " " + rsl + "\n")
                idx = idx + len(line)
                writer.writelines("\n")

        return all_token, pred_slot, real_slot, pred_intent, real_intent


class Evaluator(object):

    @staticmethod
    def intent_acc(pred_intent, real_intent):
        total_count, correct_count = 0.0, 0.0
        for p_intent, r_intent in zip(pred_intent, real_intent):

            if p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """
        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score_intents(pred_array, real_array):
        pred_array = pred_array.transpose()
        real_array = real_array.transpose()
        P, R, F1 = 0, 0, 0
        for i in range(pred_array.shape[0]):
            TP, FP, FN = 0, 0, 0
            for j in range(pred_array.shape[1]):
                if (pred_array[i][j] + real_array[i][j]) == 2:
                    TP += 1
                elif real_array[i][j] == 1 and pred_array[i][j] == 0:
                    FN += 1
                elif pred_array[i][j] == 1 and real_array[i][j] == 0:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
            P += precision
            R += recall
        P /= pred_array.shape[0]
        R /= pred_array.shape[0]
        F1 /= pred_array.shape[0]
        return F1

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
