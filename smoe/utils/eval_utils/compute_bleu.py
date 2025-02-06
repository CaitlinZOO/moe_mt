import sacrebleu
import os
import sys
import json

input_file=sys.argv[1]
lang=sys.argv[2]

def clean_text(text):
    text = text.replace('{', '')  ## 增加
    text = text.replace('}', '')  ## 增加
    text = text.replace('`','')  ## 增加
    text = text.replace("<|begin_of_text|>",'')  ## 增加
    text = text.replace("\n\n",'')  ## 增加
    text = text.replace("<|eot_id|>",'')  ## 增加
    text = text.replace("\n", "")
    text = text.replace("\n", "")
    text = text.replace("\"", "")  ## 增加
    text = text.replace("\\", "")  ## 增加
    text = text.replace("\t", "")  ## 增加
    text = text.replace("\\\\", "")  ## 增加
    text = text.strip('"') ### some translation in the fomat "{response}"
    text = text.lower()   ### 为moe增加的
    return text

hyps = []
refs = []
with open(input_file, "r") as f:
    for line in f:
        data = json.loads(line.strip())
        response = clean_text(data["pred_text"])  ## response
        reference = clean_text(data["output"])  ## reference
        if reference != "":
            hyps.append(response)
            # refs.append(data["reference"])
            refs.append(reference)

if lang == "zh" or lang == "zh-CN":
    tokenize = "zh"
elif lang == "ja":
    tokenize = "ja-mecab"
else:
    tokenize = "13a"

score = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
print(score)