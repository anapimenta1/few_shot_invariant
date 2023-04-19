data_path="/home/guests2/apa/datasets_converted"
all_sources="breakhis lung_colon_image_set nct_dataset"

for source in ${all_sources}
do
    source_path=${data_path}/${source}
    find ${source_path} -name '*.tfrecords' -type f -exec sh -c 'python3 -m tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${source_path} {} \;
done
