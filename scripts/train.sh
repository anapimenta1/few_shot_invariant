method=$1
train1="crc-tp"
train2="nct"
test="lc25000"  # Choose the desired test dataset

seeds="[2021]"
train="['$train1', '$train2']"
valid="[]"  # You can specify validation sources if needed
test="['$test']"

base_config_path="config/base.yaml"
method_config_path="config/${method}.yaml"

python3 -m src.train --base_config ${base_config_path} \
                     --method_config ${method_config_path} \
                     --opts train_sources ${train} \
                            seeds ${seeds} \
                            val_sources ${valid} \
                            test_sources ${test}
