import unicodedata
import torch

from tqdm import tqdm


class CreateDateset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data_df):

        # 誤変換と正しい文章をそれぞれ正規化し、
        # それらの間でトークン列に対応がつくもののみを抜き出す。
        data_df['wrong_text'] = data_df['wrong_text'].map(self._normalize)  # Dataflame毎に関数を適用する
        data_df['correct_text'] = data_df['correct_text'].map(self._normalize)

        return data_df[['wrong_text', 'correct_text']].to_dict(orient='records')

    def _normalize(self, text):
        """
        文字列の正規化
        """
        text = text.strip()
        text = unicodedata.normalize('NFKC', text)
        return text

class CreateDatesetLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, dataset, max_length):
        """
        データセットをデータローダに入力可能な形式にする。
        """
        dataset_for_loader = []
        for sample in tqdm(dataset):
            wrong_text = sample['wrong_text']
            correct_text = sample['correct_text']
            # print(wrong_text)
            # print(correct_text)
            encoding = self.tokenizer.encode_plus_tagged(
                wrong_text, correct_text, max_length=max_length
            )
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}
            dataset_for_loader.append(encoding)
        return dataset_for_loader
