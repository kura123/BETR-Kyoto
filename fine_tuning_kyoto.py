import time
import random
from pyknp import Juman
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from mymodule.juman_tokenizer import JumanTokenizer
from mymodule.dataset import CreateDateset, CreateDatesetLoader
import mymodule.util as util
from mymodule.train_bert import BertForMaskedLM_pl



MODEL_NAME = 'Japanese_L-12_H-768_A-12_E-30_BPE_transformers'
TRAIN_DATA_PATH = './first_train/train_data/data.txt'
TRAIN_LABEL_PATH = './first_train/train_data/label.txt'
TRAIN_VAL_RATE = 0.9
MAX_LENGTH = 128

def main(train_data_path, train_label_path):
    tokenizer = JumanTokenizer(MODEL_NAME)
    

    # データのロード
    make_data_time = time.time()
    dataseter = CreateDateset(tokenizer)
    dataset_train, dataset_val = preparation_train_data(train_data_path, train_label_path, dataseter)
    loader = CreateDatesetLoader(tokenizer)

    dataset_train_for_loader = loader(dataset_train, MAX_LENGTH)
    dataset_val_for_loader = loader(dataset_val, MAX_LENGTH)

    # データローダの作成
    # dataloader_train = DataLoader(dataset_train_for_loader, batch_size=32, shuffle=True)
    dataloader_train = DataLoader(dataset_train_for_loader, batch_size=16, shuffle=True)
    # dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)
    dataloader_val = DataLoader(dataset_val_for_loader, batch_size=32)

    end_date_time = time.time()
    print('データ作成にかかった時間', end_date_time - make_data_time)

    start_fine_time = time.time()

    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        # save_top_k=-1,  # エポックごとにモデル保存する
        save_weights_only=True,  # 重みのみ保存する
        dirpath='model/',
        filename='{epoch:02d}-{val_loss:.2f}'
    )
    trainer = pl.Trainer(
        gpus=1,  # gpuの時設定
        max_epochs=50,  # エポック数
        callbacks=[checkpoint, early_stop_callback]
    )
    # ファインチューニング
    model = BertForMaskedLM_pl(MODEL_NAME, lr=1e-5)
    trainer.fit(model, dataloader_train, dataloader_val)
    best_model_path = checkpoint.best_model_path
    end_fine_time = time.time()
    print('ファインチューニングにかかった時間', end_fine_time - start_fine_time)



def preparation_train_data(train_data_path, train_label_path, dataseter):
    data_list = util.get_file_line_data_list(train_data_path)
    label_list = util.get_file_line_data_list(train_label_path)

    assert len(data_list) == len(label_list)

    tmp_data_list = [[data, label] for data, label in zip(data_list, label_list)]    
    train_df = pd.DataFrame(tmp_data_list,
        columns=['wrong_text', 'correct_text'])
    dataset = dataseter(train_df)
    random.shuffle(dataset)

    all_data_size = len(dataset)
    train_size = int(all_data_size * TRAIN_VAL_RATE)
    dataset_train = dataset[:train_size]
    dataset_val = dataset[train_size:]

    return dataset_train, dataset_val                

if __name__ == '__main__':
    random.seed(0)
    train_data_path = TRAIN_DATA_PATH
    train_label_path = TRAIN_LABEL_PATH
    main(train_data_path, train_label_path)
