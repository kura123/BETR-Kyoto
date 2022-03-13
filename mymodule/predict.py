import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from mymodule.train_bert import BertForMaskedLM_pl


class BertPredict:
    def __init__(self, model_path, tokenizer, flag=True) -> None:
        if flag:
            self.model = BertForMaskedLM_pl.load_from_checkpoint(model_path)
        else:
            # 事前学習用
            self.config = BertConfig.from_json_file(model_path + '/config.json')
            self.model = BertForMaskedLM.from_pretrained(model_path + '/pytorch_model.bin', config=self.config)
    
        if torch.cuda.is_available():
            self.bert_mlm = self.model.bert_mlm.cuda()
        else:
            self.bert_mlm = self.model.bert_mlm.cpu()

        self.tokenizer = tokenizer

    def predict(self, text):
        # 符号化
        encoding, spans = self.tokenizer.encode_plus_untagged(text, return_tensors='pt')
    
        if torch.cuda.is_available():
            encoding = {k: v.cuda() for k, v in encoding.items()}
        else:
            encoding = {k: v for k, v in encoding.items()}

        
        with torch.no_grad():
            outputs = self.bert_mlm(**encoding)
            predictions = outputs[0]
            predicted_indexes = predictions[0].argmax(-1).cpu().numpy().tolist()
        
        # ラベル列を文章に変換
        predict_text = self.tokenizer.convert_bert_output_to_text(
            text, predicted_indexes, spans
        )
        
        return predict_text, len(predicted_indexes), len(encoding['input_ids'][0])
