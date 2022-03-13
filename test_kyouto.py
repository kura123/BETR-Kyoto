import pytorch_lightning as pl
from tqdm import tqdm
import torch
import mymodule.util as util
from mymodule.train_bert import BertForMaskedLM_pl
from mymodule.predict import BertPredict
from mymodule.juman_tokenizer import JumanTokenizer



PREMODEL_PATH = './Japanese_L-12_H-768_A-12_E-30_BPE_transformers' 


MODEL_PATH = '.model/epoch=04-val_loss=0.67.ckpt'

TEST_DATA_PATH = './test_data/data.txt'
TEST_LABEL_PATH = './test_data/label.txt'


def main(model_path, test_data_path, test_label_path):
    tokenizer = JumanTokenizer(PREMODEL_PATH)

    test_data_list = util.get_file_line_data_list(test_data_path)
    test_label_list = util.get_file_line_data_list(test_label_path)

    bert_predicter = BertPredict(model_path, tokenizer)
    for data, label in zip(test_data_list, test_label_list):
        predict_text, predict_token_len, data_token_len = bert_predicter.predict(data)  # BERTによる予測
        pre_label = tokenizer.preparation_text(label)
        (true_text, prediction_text, hantei, seido) = util.seido_calculation(predict_text, pre_label)
        print('%s｜%s｜%s｜%s' % (true_text, prediction_text, hantei, seido))


if __name__ == '__main__':
    model_path = MODEL_PATH
    test_data_path = TEST_DATA_PATH
    test_label_path = TEST_LABEL_PATH
    main(model_path, test_data_path, test_label_path)
