import os
import sys
import pprint
import argparse


import json
from retrieval import information_retrieval
from qa import QaModule, print_answers_in_file, rankAnswers, rankAnswersList
pp = pprint.PrettyPrinter(indent=4)


def information_retrieval_fake(file_path):
    # since we don't need a information retrieval function this time, this function will just format the input data to the format
    all_results = []
    data_for_qa = []
    f = open(file_path, 'r')   # debatepedia.source
    for line in f.readlines():
        data = {}
        data["context"] = []
        line = line.strip()
        question = line.split('[SEP]')[-1]
        context = line.split('[SEP]')[0]
        data["answer"] = ""
        data["context"].append(context)
        data["doi"] = [""]
        data["titles"] = [""]
        data_for_qa.append({"question": question, "data": data})
    f.close()
    return all_results, data_for_qa

# tf 1.15.2  ["/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1586435240", "/home/xuyan/kaggle/bioasq-biobert/model/1586435317"]
# tf 1.13.1  ["/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1564469515", "/home/xuyan/kaggle/bioasq-biobert/model/1585470591"]
# qa_model = QaModule(['mrqa','biobert'], ["/home/farhad/covid19data/qa_models/1564469515", "/home/farhad/covid19data/qa_models/1585470591"], "/home/farhad/covid19data/qa_models/spiece.model", "/home/farhad/covid19data/qa_models/bert_config.json", "/home/farhad/covid19data/qa_models/vocab.txt")
# qa_model = QaModule(['mrqa'], ["/home/farhad/covid19data/qa_models/1564469515"], "/home/farhad/covid19data/qa_models/spiece.model", "/home/farhad/covid19data/qa_models/bert_config.json", "/home/farhad/covid19data/qa_models/vocab.txt")
# qa_model = QaModule(['mrqa'], ["/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1596112521"], "/home/farhad/covid19data/qa_models/spiece.model", "/home/farhad/covid19data/qa_models/bert_config.json", "/home/farhad/covid19data/qa_models/vocab.txt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ans_prob_output_file_path", type=str, default='/home/sudan/AAAI2021/QFS_QA/Debatepedia_Output/valid.ans_prob')
    parser.add_argument("--file_source", type=str, default='/home/sudan/Kaggle/bart/data/Debatepedia/data_fold/1/valid.source')
    parser.add_argument("--model_path", type=str, default="/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1596112521")
    parser.add_argument("--spiece_model", type=str, default="/home/farhad/covid19data/qa_models/spiece.model")
    parser.add_argument("--bert_config", type=str, default="/home/farhad/covid19data/qa_models/bert_config.json")
    parser.add_argument("--bert_vocab", type=str, default="/home/farhad/covid19data/qa_models/vocab.txt")
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()

    ans_prob_output_file_path = args.ans_prob_output_file_path
    file_source = args.file_source
    split_flag = args.split

    all_results, data_for_qa = information_retrieval_fake(file_source)
    qa_model = QaModule(['mrqa'], [args.model_path], args.spiece_model, args.bert_config, args.bert_vocab)

    print("Get Answers...")
    answers = qa_model.getAnswers(data_for_qa, ans_prob_output_file_path)
    format_answer = qa_model.makeFormatAnswersList(answers)

    # save the original answer to json file
    output_dir = '/'.join(ans_prob_output_file_path.split('/')[:-1])
    qa_ans_output_json_file = output_dir + '/debatepedia_QA_original_mrqa_only_' + split_flag + '.json'
    context_sep_question_file = output_dir + '/debatepedia_QA_context_question_' + split_flag + '_mrqa_only.source' # "./Debatepedia_Output/debatepedia_QA_context_question_valid_mrqa_only.source"
    context_ans_sep_question_file = output_dir + '/debatepedia_QA_con_ansraw_question_' + split_flag + '_mrqa_only.source'    # "./Debatepedia_Output/debatepedia_QA_con_ansraw_question_valid_mrqa_only.source"


    with open(qa_ans_output_json_file, "w") as f:
        json.dump(format_answer, f)

    # context + [SEP] + question
    with open(context_sep_question_file, "w") as f:
        for each_0 in format_answer:
            for each in each_0:
                line = each["question"] + " [SEP] "
                for ans in each['raw']:
                    line += ans
                    line += ". "
                line += each["context"]
                f.write("%s\n" % line)

    # context + " " + answer + [SEP] + question
    with open(context_ans_sep_question_file, "w") as f:
        for each_0 in format_answer:
            for each in each_0:
                line = each["context"] + " "
                for ans in each['raw']:
                    line += ans
                    line += ". "
                line += " [SEP] "
                line += each["question"]
                f.write("%s\n" % line)
