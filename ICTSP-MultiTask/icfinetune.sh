if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

log_folder=Finetune

if [ ! -d "./logs/"$log_folder ]; then
    mkdir ./logs/$log_folder
fi

resume="resume.pth"
icpretrain_config_path="./configs/pretrain_configs_finetune.json"

root_path_name=./dataset_test/
features=M
data_name="ETTh2.csv" # only for validation and testing
data_type=ETTh2

patience=200
seq_len=2048
pred_len=96
random_seed=2024
test_every=20
plot_every=1
plot_full_details=0
scale=1
iterative_prediction=0

model_name=ICTSPretrain
batch_size=8
gradient_accumulation=16
batch_size_test=8
learning_rate=0.00001
max_grad_norm=1

train_ratio=0.7
test_ratio=0.2

number_of_targets=0

python -u run_longExp.py \
  --is_training 1 \
  --model_type 'IC' \
  --root_path $root_path_name \
  --data_path $data_name \
  --model_id $model_name'_'$random_seed'_'$data_name'_'$data_type'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_type \
  --features $features \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --scale $scale \
  --des 'Exp' \
  --patience $patience \
  --test_every $test_every \
  --random_seed $random_seed \
  --max_grad_norm $max_grad_norm \
  --train_epochs 2000 \
  --itr 1 \
  --batch_size $batch_size \
  --batch_size_test $batch_size_test \
  --plot_every $plot_every \
  --learning_rate $learning_rate \
  --plot_full_details $plot_full_details \
  --devices "0,1" \
  --num_workers 6 \
  --resume $resume \
  --use_amp \
  --train_ratio $train_ratio \
  --test_ratio $test_ratio \
  --icpretrain_config_path $icpretrain_config_path \
  --number_of_targets $number_of_targets \
  --iterative_prediction $iterative_prediction >logs/$log_folder'/'$model_name'_'$random_seed'_'$data_name'_'$data_type'_'$seq_len'_'$pred_len