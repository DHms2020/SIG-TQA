#!/usr/bin/env python

import _jsonnet
import json
import argparse
import collections
import attr
from seq2struct.commands import preprocess, train, infer, eval, gen
import crash_on_ipy
import os
import sys

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()

@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    local_rank = attr.ib()

@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)

@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()

@attr.s
class GenConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    schema_path=attr.ib()
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval")
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    parser.add_argument("--local_rank",default=os.getenv('LOCAL_RANK',-1),type=int)
    args = parser.parse_args()
    
    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = json.dumps(exp_config["model_config_args"])
    else:
        model_config_args = None
    
    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, \
            model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file, 
            model_config_args, exp_config["logdir"],args.local_rank)
        train.main(train_config)
    elif args.mode == "eval":
        for step in exp_config["eval_steps"]:
            infer_output_path = "{}/{}-step{}.infer".format(
                exp_config["eval_output"], 
                exp_config["eval_name"], 
                step)
            infer_config = InferConfig(
                model_config_file,
                model_config_args,
                exp_config["logdir"],
                exp_config["eval_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                use_heuristic=exp_config["eval_use_heuristic"]
            )
            infer.main(infer_config)

            eval_output_path = "{}/{}-step{}.eval".format(
                exp_config["eval_output"], 
                exp_config["eval_name"], 
                step)
            eval_config = EvalConfig(
                model_config_file,
                model_config_args,
                exp_config["logdir"],
                exp_config["eval_section"],
                infer_output_path,
                eval_output_path
            )
            eval.main(eval_config)

            res_json = json.load(open(eval_output_path))
            print(step, res_json['total_scores']['all']['exact'])

    elif args.mode == 'gen':
        for step in exp_config["eval_steps"]:
            ABSPATH=os.path.dirname(os.path.abspath(__file__))
            schema_path=os.path.join(ABSPATH,'data/Spider/tables.json')
            # #迁移到DuSQL中文数据集下
            # schema_path=os.path.join(ABSPATH,'data/DuSQL/DuSQL_schema.json')
            infer_output_path = "{}/{}-step{}.infer".format(
                exp_config["eval_output"],
                exp_config["eval_name"],
                step)
            gen_config = GenConfig(
                model_config_file,
                model_config_args,
                exp_config["logdir"],
                exp_config["gen_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                schema_path,
                use_heuristic=exp_config["eval_use_heuristic"]
            )
            gen.main(gen_config)


if __name__ == "__main__":
    main()