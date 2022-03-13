import re
import torch
import unicodedata
import mojimoji
from pyknp import Juman
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

class JumanTokenizer:
    def __init__(self, model_file) -> None:
        self.model_file = model_file
        self.juman = Juman()
        self.tokenizer = BertTokenizer.from_pretrained(self.model_file)

    
    def tokenize_juman(self, text):

        word_list = self.juman_exe(text)

        tokenized_text = []
        for word in word_list:
            tokenized_text.extend(self.tokenizer.tokenize(word))

        tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([tokens])
        
        return tokens_tensor
    
    def encode_plus_tagged(self, wrong_text, correct_text, max_length=128):
        """
        ファインチューニング時に使用。
        誤変換を含む文章と正しい文章を入力とし、
        符号化を行いBERTに入力できる形式にする。
        """
        wrong_word_list = self.juman_exe(wrong_text)
        # 誤変換した文章をトークン化し、符号化
        encoding = self.tokenizer(
            wrong_word_list,
            padding='max_length', truncation=True, 
            max_length=max_length, is_split_into_words=True
        )
        # 正しい文章をトークン化し、符号化
        correct_word_list = self.juman_exe(correct_text)
        encoding_correct = self.tokenizer(
            correct_word_list,
            padding='max_length', truncation=True, 
            max_length=max_length, is_split_into_words=True
        )

        # 正しい文章の符号をラベルとする
        encoding['labels'] = encoding_correct['input_ids']

        return encoding
    
    def juman_exe(self, text):
        pre_text = self.preparation_text(text)
        juman_result = self.juman.analysis(pre_text)
        word_list = [mrph.midasi for mrph in juman_result.mrph_list()]

        return word_list
    
    def preparation_text(self, text):
        '''juman++が半角に対応していないために、半角文字を全角に変換'''
        ret_text = text.replace(' ', '　')
        ret_text = mojimoji.han_to_zen(ret_text)
        return ret_text
    
    def encode_plus_untagged(self, text, max_length=None, return_tensors=None):
        """
        文章を符号化し、それぞれのトークンの文章中の位置も特定しておく。
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = []  # トークンを追加していく。
        tokens_original = []  # トークンに対応する文章中の文字列を追加していく。
        pre_text = self.preparation_text(text)
        word_list = self.juman_exe(text)
        for word in word_list:
            # 単語をサブワードに分割
            tokens_word = self.tokenizer.tokenize(word)
            tokens.extend(tokens_word)
            if len(tokens_word) == 0:
                # \u300（全角空白）のための対応
                continue
            elif tokens_word[0] == '[UNK]':
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##', '') for token in tokens_word
                ])
            
        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            lenght = len(token)
            while 1:
                if token != pre_text[position:position + lenght]:
                    position += 1
                else:
                    spans.append([position, position + lenght])
                    position += lenght
                    break
        
        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        encoding = self.tokenizer.prepare_for_model(
            input_ids,
            max_length=max_length,
            padding='max_length' if max_length else False,
            truncation=True if max_length else False
        )

        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length - 2]
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * (sequence_length - len(spans))

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = {k: torch.tensor([v]) for k, v in encoding.items()}

        return encoding, spans

    def convert_bert_output_to_text(self, text, labels, spans):
        """
        推論時に使用。
        文章と、各トークンのラベルの予測値、文章中での位置を入力とする。
        そこから、BERTによって予測された文章に変換。
        """
        assert len(spans) == len(labels)
        pre_text = self.preparation_text(text)

        # labels, spansから特殊トークンに対応する部分を取り除く
        labels = [label for label, span in zip(labels, spans) if span[0] != -1]
        spans = [span for span in spans if span[0] != -1]

        # BERTが予測した文章を作成
        predicted_text = ''
        position = 0
        for label, span in zip(labels, spans):
            start, end = span
            if position != start:  # 空白の処理
                predicted_text += pre_text[position:start]
            predicted_token = self.tokenizer.convert_ids_to_tokens(label)
            predicted_token = predicted_token.replace('##', '')
            predicted_text += predicted_token
            position = end
        
        # 特殊トークンは、空文字に変換
        ret_predicted_text = re.sub(r'\[SEP\]|\[PAD\]|\[UNK\]', '', predicted_text)
        return ret_predicted_text
