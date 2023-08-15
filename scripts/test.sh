method=$1
shot=$2
train=$3
valid=$4
test=$5


visu="False"

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

seeds="[2021]"
train="['$train']"
valid="['$valid']"
test="['$test']"

python3 -m src.test --base_config ${base_config_path} \
                    --method_config ${method_config_path} \
                    --opts num_support ${shot} \
                           train_sources ${train} \
                           seeds ${seeds} \
                           visu ${visu} \
                           val_sources ${valid} \
                           test_sources ${test} \
                            # | tee ${dirname}/log_${method}.txt
