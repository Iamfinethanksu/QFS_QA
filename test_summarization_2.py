import os
import sys
import pprint

import json


with open("debatepedia_QA_original_test.json", "r") as f:
    format_answer = json.load(f)

with open("debatepedia_QA_test_answer.source", "w") as f:
    for each_0 in format_answer:
        for each in each_0:
            line = each["question"].strip() + " [SEP] "
            line += each['answer']   # ans_sent
            line += ". "
            line += each["context"]
            f.write("%s\n" % line)

with open("debatepedia_QA_test_answer.source.1", "w") as f:
    for each_0 in format_answer:
        for each in each_0:
            line = each["context"] + " "
            line += each['answer']
            line += ". [SEP] "
            line += each["question"].strip()
            f.write("%s\n" % line)