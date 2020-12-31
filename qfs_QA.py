import os
import sys
import pprint
import argparse


import json
# from retrieval import information_retrieval
from qa import QaModule, print_answers_in_file, rankAnswers, rankAnswersList
pp = pprint.PrettyPrinter(indent=4)


def read_json(name):
    file = open(name)
    data = file.readline()
    json_data = json.loads(json.dumps(eval(data)))
    return json_data

def read_debatepedia_data(file_path):
    # since we don't need a information retrieval function this time, this function will just format the input data to the format
    data_for_qa = []
    # debatepedia.source
    f = open(file_path, 'r')   
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
    return data_for_qa

def read_duc_data(file_path):
    data_duc = {}
    data_for_qa = []
 
    context_file = file_path + '/paragraphs_400.txt'
    query_file = file_path + '/topics.txt'
    context_data = read_json(context_file)
    query_data = read_json(query_file)

    assert len(context_data.keys()) == len(query_data.keys()), 'the length of the paragraphs file is not equal to the topics file'
    for key, contexts in context_data.items():
        question = query_data[key]
        for context in contexts:
            data = {}
            data["context"] = []
            data["answer"] = ""
            data["doi"] = [""]
            data["titles"] = [""]
            if context == "":
                print("the context is empty")
            data["context"].append(context)
            data_for_qa.append({"question": question, "data": data})
    data_duc[key]=data_for_qa

    return data_duc
    

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
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--data_format", type=str, default="debatepedia")

    args = parser.parse_args()

    ans_prob_output_file_path = args.ans_prob_output_file_path
    file_source = args.file_source
    split_flag = args.split
    data_format = args.data_format
    output_dir = '/'.join(ans_prob_output_file_path.split('/')[:-1])

    print('the output_dir is {}'.format(output_dir))
    if not os.path.isdir(output_dir):
        print("exit since the output directory {} does not exist".format(output_dir))
        sys.exit(0)


    if data_format == 'debatepedia':
        data_for_qa = read_debatepedia_data(file_source)
        qa_model = QaModule(['mrqa'], [args.model_path], args.spiece_model, args.bert_config, args.bert_vocab)
        print("Get Answers...")
        answers, ans_relevance_prob_lines = qa_model.getAnswers(data_for_qa)
        format_answer = qa_model.makeFormatAnswersList(answers)

        print("there are all {} lines have been processed".format(len(ans_relevance_prob_lines)))
        with open(ans_prob_output_file_path, 'w') as fout:
            for line in ans_relevance_prob_lines:
                fout.write(line)

        # save the original answer to json file
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
    
    elif data_format == 'duc':
        qa_ans_output_data = {}
        ans_relevance_prob_data = {}
        data_duc = read_duc_data(file_source)
        qa_model = QaModule(['mrqa'], [args.model_path], args.spiece_model, args.bert_config, args.bert_vocab)
        for key, data_for_qa in data_duc.items():
            print("Get Answers...")
            answers, ans_relevance_prob_lines = qa_model.getAnswers(data_for_qa)
            format_answer = qa_model.makeFormatAnswersList(answers)
            qa_ans_output_data[key] = format_answer
            ans_relevance_prob_data[key] = ans_relevance_prob_lines

        qa_ans_output_json_file = output_dir + '/original.json' 
        with open(qa_ans_output_json_file, "w") as f:
            json.dump(qa_ans_output_data, f)
        
        with open(ans_prob_output_file_path, 'w') as fout:
            json.dump(ans_relevance_prob_data, fout)



        








