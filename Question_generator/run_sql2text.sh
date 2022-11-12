#!/bin/bash

export PYTHON_HOME=/opt/conda/envs/py3.7/
export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}
export PATH=${PYTHON_HOME}/bin:${PATH}

export PYTHONPATH=./:${PYTHONPATH:-}


if [ -d output ]; then
rm -r output
fi
mkdir output
mkdir output/log

model_save_path=./output
tensor_board_path=./output/log

infer_model_path=$(pwd)/infermodel/

echo "----------Begin--Training-----------"
nohup python relogic/sql-to-text-train.py \
      --tokenizer_name relogic/bert-base-uncased/ \
      --train_data_file "Spider_sql2text_train.json" \
      --eval_data_file "Spider_sql2text_val.json" \
      --output_dir "${model_save_path}" \
      --logging_dir "${tensor_board_path}" \
      --do_train \
      --do_eval \
      --num_train_epochs 20 \
      --save_steps 10000 \
      --eval_steps 500 \
      --evaluate_during_training \
      &

# for inference  setting:
#      --do_generate \
#      --model_name_or_path  "${infer_model_path}" \
#      --generate_data_file "sql2text_input.json" \

