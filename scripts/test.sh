method=$1
shot=$2
train1=$3
train2=$4
valid=$5
test=$6


visu="False"

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

seeds="[2021]"
train1="['$train1']"
train2="['$train2']"
valid="['$valid']"
test="['$test']"

python3 -m src.test --base_config ${base_config_path} \
                    --method_config ${method_config_path} \
                    --opts num_support ${shot} \
                           train_source_1 ${train1} \
                           train_source_2 ${train2} \
                           seeds ${seeds} \
                           visu ${visu} \
                           val_sources ${valid} \
                           test_sources ${test} \
                            # | tee ${dirname}/log_${method}.txt
