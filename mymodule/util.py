import Levenshtein
from mojimoji import zen_to_han

def seido_calculation(prediction_text, true_text):
    """手書き文字の精度計算関数
       一文字単位の精度は、（正解の文字数 - 編集距離）/ (正解の文字数)で計算を行う

    Args:
        prediction_text ([str]): 予測テキスト
        true_text ([str]): 正解テキスト

    Returns:
        [tuple]: 精度計算結果
                 (正解テキスト, 予測テキスト, 正解不正解, 一文字単位の精度)
    """
    prediction_text = zen_to_han(prediction_text, kana=False).replace(' ', '')
    true_text = zen_to_han(true_text, kana=False).replace(' ', '')
    dis = Levenshtein.distance(prediction_text, true_text)
    seido = (max(len(true_text), len(prediction_text)) - dis) / max(len(true_text), len(prediction_text))
    hantei = prediction_text == true_text
    return (true_text, prediction_text, hantei, seido)


def get_file_line_data_list(path, encode='utf_8'):
    with open(path, encoding=encode) as f:
        l_strip = [s.strip() for s in f.readlines()]
    return l_strip
