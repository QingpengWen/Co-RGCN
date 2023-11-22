# -*- coding: utf-8 -*-#
"""
@CreateTime :       2023/2/28 21:25
@Author     :       Qingpeng Wen
@File       :       train.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2023/3/15 23:35
"""

import os, json, random
import numpy as np
import torch.optim as optim
from models.module import ModelManager
from utils.loader import DatasetManager
from utils.process import Processor
from utils.config import *
import fitlog
import warnings
warnings.filterwarnings("ignore")

model_file_path = r"sss"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if __name__ == "__main__":
    fitlog.set_log_dir("logs/")
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    args = parser.parse_args()

    if not args.do_evaluation:
        # Save training and model parameters.
        if not os.path.exists(args.save_dir):
            os.system("mkdir -p " + args.save_dir)

        log_path = os.path.join(args.save_dir, "param.json")
        with open(log_path, "w") as fw:
            fw.write(json.dumps(args.__dict__, indent=True))
        # Fix the random seed of package random.
        random.seed(args.random_state)
        np.random.seed(args.random_state)
        # Fix the random seed of Pytorch when using GPU.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_state)
            torch.cuda.manual_seed(args.random_state)

        # Fix the random seed of Pytorch when using CPU.
        torch.manual_seed(args.random_state)
        torch.random.manual_seed(args.random_state)
        # Load pre-training model

        if os.path.exists(model_file_path):
            checkpoint = torch.load(model_file_path, map_location=device)
            model = checkpoint['model']
            dataset = DatasetManager(args)
            dataset.quick_build()
            optimizer = optim.Adam(model.parameters(), lr=dataset.learning_rate, weight_decay=dataset.l2_penalty)
            start_epoch = checkpoint["epoch"]
            dataset.show_summary()
            model.show_summary()
            process = Processor(dataset, model, optimizer, start_epoch,  args.batch_size, args)
            print('epoch {}: The pre-training model was successfully loaded！'.format(start_epoch))
        else:
            # Instantiate a dataset object.
            print('No save model will be trained from scratch！')
            start_epoch = 0
            dataset = DatasetManager(args)
            dataset.quick_build()
            dataset.show_summary()
            model_fn = ModelManager
            # Instantiate a network model object.
            model = ModelManager(
                args, len(dataset.word_alphabet),
                len(dataset.slot_alphabet),
                len(dataset.intent_alphabet)
            )
            model.show_summary()
            optimizer = optim.Adam(model.parameters(), lr=dataset.learning_rate, weight_decay=dataset.l2_penalty)

            process = Processor(dataset, model, optimizer, start_epoch, args.batch_size, args)
        try:
            process.train()
        except KeyboardInterrupt:
            print("Exiting from training early.")

    if not args.do_evaluation:
        model = torch.load(os.path.join(args.save_dir, "model/model.pkl"))
        dataset = torch.load(os.path.join(args.save_dir, "model/dataset.pkl"))
        checkpoint = torch.load(os.path.join(args.save_dir, "model/model_epoch.pkl"), map_location=device)
    else:
        print("Beginning evaluation:")
        model = torch.load(os.path.join(args.save_dir, "model/model.pkl"))
        dataset = torch.load(os.path.join(args.save_dir, "model/dataset.pkl"))

    result = Processor.validate(
        os.path.join(args.save_dir, "model/model.pkl"),
        dataset,
        args.batch_size, len(dataset.intent_alphabet), args=args)

    print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
