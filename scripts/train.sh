method=$1
train1=$2
train2=$3
valid=$4
test=$5

seeds="[2021]"
train1="['$train1']"
train2="['$train2']"
valid="['$valid']"
test="['$test']"

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

python3 -m src.train --base_config ${base_config_path} \
                     --method_config ${method_config_path} \
                     --opts train_source_1 ${train1} \
                            train_source_2 ${train2} \
                            seeds ${seeds} \
                            val_sources ${valid} \
                            test_sources ${test} \
