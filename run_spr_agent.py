# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

import json
from datetime import datetime, timedelta

import json
from tqdm import tqdm
import os
import traceback
import random
from global_config import (
    LLM_MODEL_NAME,
    DO_REFERENCE_
    DO_FUSION_JUDGE,
    FUSION_TEMP,
    SEARCH_ROUTE,
)
import sys
import glob
from utils import get_md5
import shutil
from pipeline_spar import AcademicSearchTree

file_lst = [
    "./global_config.py",
    "./instruction.py",
    "./run_spr_agent.py",
    "./search_engine.py",
    "./api_web.py",
    "./pipeline_spar.py"
]

sample_num = 2000
score_thresh = 0.5
max_depth = 2
relevance_doc_num = 10

benchmark_map = {
    "AutoScholarQuery": {
        "src_file": "./benchmark/AutoScholarQuery_test.jsonl",
        "select_file": f"./benchmark/AutoScholarQuery_test_select_{sample_num}.jsonl",
    },
    "OwnBenchmark": {
        "src_file": "./benchmark/spar_bench.jsonl",
        "select_file": f"code_official/benchmark/spar_bench_select_{sample_num}.jsonl"

    },
}

benchmark_name = "AutoScholarQuery"
# benchmark_name = "OwnBenchmark"

benchmark_name = sys.argv[1]

src_file = benchmark_map[benchmark_name]["src_file"]
select_file = benchmark_map[benchmark_name]["select_file"]
print(f"select_file: {select_file}")


output_folder = f"./gen_result/{benchmark_name}_{sample_num}_msearch_{'-'.join(SEARCH_ROUTE)}_depth{max_depth}_do_reference_{DO_REFERENCE_SEARCH}_query_judge_{DO_FUSION_JUDGE}_fusion_{FUSION_TEMP}_no_enddate_no_autocorrect_pasa_score_{score_thresh}"  # 加上query fusion

print(f"output_folder: {output_folder}")

os.makedirs(output_folder, exist_ok=True)


search_agent = AcademicSearchTree(
    max_depth=max_depth, max_docs=relevance_doc_num, similarity_threshold=score_thresh
)


for one in file_lst:
    shutil.copy2(one, output_folder)

already = {}
for one in glob.glob(f"{output_folder}/*.json"):
    with open(one, "r") as fr:
        info = json.load(fr)
    question = info["search_query"]
    already[question] = one

with open(src_file, "r") as f:
    if src_file.endswith(".jsonl"):
        lines = f.readlines()
        random.seed(123)
        random.shuffle(lines)
        lines = lines[:sample_num]
    elif src_file.endswith(".json"):
        lines = json.load(f)
    print(f"lines: {len(lines)}")

    with open(select_file,"w") as fw:
        for one in lines:
            fw.write(one.strip() + "\n")


    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Processing lines"):
        try:
            if isinstance(line, str):
                data = json.loads(line)
                question = data["question"]
            elif isinstance(line, dict):
                data = line
                question = data["query"]
            else:
                data = {}
                question = line

            end_date = ""
            if question in already:
                print(f"pass: {already[question]}")
                continue

            dest_name = get_md5(question)
            dest_file = os.path.join(output_folder, f"{dest_name}.json")

            sorted_docs = search_agent.search(question, end_date=end_date)

            if "answer" in data:
                search_agent.root.extra["answer"] = data["answer"]

            elif "data_result_add_score" in data:
                search_agent.root.extra["answer"] = [
                    one["title"] for one in data["data_result_add_score"]
                ]

            if output_folder != "":
                res = search_agent.root.convert_to_dict()
                with open(dest_file, "w") as fw:
                    json.dump(res, fw, indent=2)

            try:
                print("draw search tree")
                search_agent.visualize_tree(f"{output_folder}/{dest_name}")
            except:
                traceback.print_exc()
                pass

            # break

        except:
            traceback.print_exc()
            pass

print(f"output_folder: {output_folder}")
