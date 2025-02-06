import sacrebleu
from tqdm import tqdm
import os
import json
import jieba
import nltk
# import MeCab
# from konlpy.tag import Mecab as KoMecab
# from pythainlp.tokenize import word_tokenize as th_word_tokenize
from nltk.translate.meteor_score import meteor_score
import re
# from bert_score import BERTScorer
# import qianfan
# 下载 wordnet 词库（只需要执行一次）
# print("Downloading wordnet...")
# nltk.download('wordnet')
# print("Done.")
# os.environ["QIANFAN_ACCESS_KEY"] = os.getenv("Q_AK")
# os.environ["QIANFAN_SECRET_KEY"] = os.getenv("Q_SK")

# def qf_get(text, chat_comp):

#     # model_name = model.model_name
#     model  = "ERNIE-4.0-8K"
#     prompt = f'''你是一个多语言翻译机器人，你的任务是给定“一个大模型对一段文字的翻译结果的输出”，根据大模型返回的本文，进行提炼总结，并输出你认为大模型对原有文字的翻译内容。
# ###要求：
# ## 1. 请不要在任何情况下添加任何前缀或后缀到原文上。
# ## 2. 请注意，上述内容是根据给定文本的内容分析翻译的结果，务必直接输出最后的结果，务必不需要额外的解释和说明。
# 大模型的原始输出的内容如下：
# ```{text}```'''
#     messages = [{
#         "role": "user",
#         "content": prompt
#     }]
#     # print(messages)
#     response = chat_comp.do(
#         model       = model,
#         temperature = 0.2,
#         max_output_tokens = 200,
#         # extra_parameters = {},
#         # response_format = 'json_object',
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     # print(response.body['result'])
#     return response.body['result'].replace('"', '')
# 初始化分词器
# print('Initializing the tokenizers...')
# mecab_ja = MeCab.Tagger()  # 日语分词
# mecab_ko = KoMecab()       # 韩语分词
# print('Done.')
# 多语言分词函数
def tokenize_text(text, lang):
    if lang == 'zh':  # 中文
        return list(jieba.cut(text))
    # elif lang == 'ja':  # 日语
    #     return mecab_ja.parse(text).strip().split()
    # elif lang == 'ko':  # 韩语
    #     return mecab_ko.morphs(text)
    # elif lang == 'th':  # 泰语
    #     return th_word_tokenize(text)
    if lang in ['ar', 'tr', 'hi', 'ru', 'es', 'fr', 'de', 'it', 'en', 'pt']:  # 阿拉伯语、土耳其语、印地语、俄语、西班牙语、法语、德语、意大利语、英语
        if isinstance(text, str):
            return text.split()
        # return text.split()
        else:
            return text
    else:
        raise ValueError(f"Unsupported language: {lang}")
def calculate_meteor(reference, hypothesis, lang):
    # 分词处理
    reference_tokens = tokenize_text(reference, lang)
    hypothesis_tokens = tokenize_text(hypothesis, lang)
    
    # 计算 METEOR 分数
    meteor = meteor_score([reference_tokens], hypothesis_tokens)
    
    return meteor        
def initialize_bertscorer(lang='en'):
    """
    初始化 BERTScorer 模型以提高批量处理性能。
    
    参数:
        lang (str): 语言代码，'en' 为英文，'zh' 为中文
    
    返回:
        scorer (BERTScorer): 已初始化的 BERTScorer 对象
    """
    scorer = BERTScorer(lang=lang)
    return scorer    
# scorer_zh = initialize_bertscorer(lang='zh')
# scorer_en = initialize_bertscorer(lang='en')
def compute_bertscore(reference, candidate, tgt_lang):
    references = [reference]
    candidates = [candidate]
    

    if tgt_lang == 'zh':
        scorer = scorer_zh
    else:
        scorer = scorer_en
    # 计算 BERTScore
    P, R, F1 = scorer.score(candidates, references)
    return  F1.item()

def calculate_metrics(reference, hypothesis, tgt_lang = 'en'):

    meteor = calculate_meteor(reference, hypothesis, tgt_lang)
    # bert_score = compute_bertscore(reference, hypothesis, tgt_lang)

    # 准备评估数据
    references = [[reference]]  # 这里的 reference 需要再加一层列表包装，表示多句子结构
    hypotheses = [hypothesis]

    # 根据目标语言选择是否使用中文分词
    bleu = sacrebleu.corpus_bleu(hypotheses, references, tokenize='zh' if tgt_lang == 'zh' else '13a')
    # ter_score = sacrebleu.metrics.TER().corpus_score(hypotheses, references)
    # chrf = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)
    
    # meteor_score的调用：参考译文是单一参考列表的列表
    # meteor = meteor_score([reference], hypothesis)  # meteor_score 只需要传入单一列表的引用

    # return bleu, ter_score, chrf, meteor


    return bleu, meteor, 
 
def get_eb_answer(text, model):
    return text
def main():

    # models = ['llama3','qwen2','fuxi']
    models = [  
        # 'llama3',
        # 'llama3_lora',
        # 'llama3-8b',
        # 'qwen0.5',
        # 'qwen1.5',
        'llama3_mixtral2group',
    ]
    total = 20962*len(models)
    pbar = tqdm(total=int(total), desc='Progress', unit='item')
    for llm in models:
        # dir_path = f'../baseline/output/{llm}/'
        dir_path = f'/home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/'
        # print(dir_path)
        # continue
        # if llm in ['deepseek', 'internvl']:
        #     chat_comp = qianfan.ChatCompletion()

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.jsonl'):
                # if file in ['ECOIT.test.jsonl','IIMT.test.jsonl','OCRMT30K.test.jsonl','Dota.test.jsonl']:
                    file_path = os.path.join(root, file)
                    # print('===', file_path, '===')

                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f.readlines():
                            item = json.loads(line)
                            item['filename'] = file.replace('.jsonl', '')
                            answer =item['pred_text']
                            # print(item)
                            # breako
                            # if llm in ['ours']:
                            #     answer = re.findall(r'<t>(.*?)</t>', item['pred_text'])
                            #     if len(answer) == 0:
                                   
                            # else:
                            #     answer = qf_get(item['pred_text'].replace('\n', ' '), chat_comp) 
                                # continue
                            # print('===========')
                            # print('SRC:', item['pred_text'].replace('\n', ' '))
                            # print('ANS:', answer)
                            hypothesis = answer
                            if not isinstance(hypothesis, str):
                                hypothesis = ' '.join(hypothesis)
    
                            # hypothesis = re.sub(r'[^\w\s]', ' ', hypothesis)
                            # hypothesis = re.sub(r'\s+', ' ', hypothesis)
                            hypothesis = hypothesis.replace('{', '').replace('}', '').replace('`','').replace("\n",' ').lower().strip()
                            reference  = item['output'].replace("\n",' ').lower()
                            # tgt_lang   = 'en'
                            # print('REF:', reference)

                            # reference_file = os.path.join(root, 'ref', file[:-5] + '.txt')
                            bleu, meteor  = calculate_metrics(reference, hypothesis)
                            # if reference == hypothesis:
                            #     bleu.score = 100
                            # print('BLEU:', bleu.score, '\tChrF++:', chrf.score)
                            print(llm, item['filename'], item['id'], bleu.score, meteor * 100)
                            pbar.update(1)# break

    pbar.close()

if __name__ == "__main__":
    main()