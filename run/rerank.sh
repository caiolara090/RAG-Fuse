# overrides
data=REINVENTA
model=RetrieverRoBERTa

text_max_length=128
label_max_length=128
label_enhancement=LLM
text_features_source=TXT

# aggregate
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python3 main.py \
    tasks=[rerank] \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=16 \
    data.num_workers=6 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/aggregate_LLM_${model}_${data}_${fold_idx}.tmr
done