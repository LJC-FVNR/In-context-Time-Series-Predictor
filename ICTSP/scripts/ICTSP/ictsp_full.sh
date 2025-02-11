if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

root_path_name=./dataset/
features=M

resume="none"
fix_embedding=1
number_of_targets=0

patience=30
seq_len=1440
lookback=512
random_seed=2024
test_every=200
plot_every=10
scale=1

model_name=ICTSP
batch_size=32
batch_size_test=32
gradient_accumulation=1
learning_rate=0.0005
# Training
time_emb_dim=0  # no timestamp embedding used
e_layers=3
d_model=128
n_heads=8
mlp_ratio=4
dropout=0.5
sampling_step=8
token_retriever_flag=1
linear_warmup_steps=5000
token_limit=2048
# Training
max_grad_norm=0

# m1

for pred_len in 96 192 336 720
do

from_ds="ETTm2.csv"
to_ds="ETTm2.csv"
enc_in=7

data_alias="Full-$from_ds-to-$to_ds-$pred_len"
transfer_learning=1
data_name="$from_ds,$to_ds" # same source and target datasets, not transfer learning
data_type=custom

python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_name \
  --model_id $model_name'_'$data_alias'_'$random_seed'_RD('$random_drop_training')_'$seq_len'_'$lookback'_'$pred_len'_K('$e_layers')_d('$d_model')_Dropout('$dropout')_m('$sampling_step')_W('$linear_warmup_steps')_Limit('$token_limit')_wRet('$token_retriever_flag \
  --model $model_name \
  --data $data_type \
  --features $features \
  --seq_len $seq_len \
  --lookback $lookback \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --number_of_targets $number_of_targets \
  --des 'Exp' \
  --patience $patience \
  --test_every $test_every \
  --time_emb_dim $time_emb_dim \
  --e_layers $e_layers \
  --d_model $d_model \
  --n_heads $n_heads \
  --mlp_ratio $mlp_ratio \
  --dropout $dropout \
  --sampling_step $sampling_step \
  --token_retriever_flag $token_retriever_flag \
  --linear_warmup_steps $linear_warmup_steps \
  --token_limit $token_limit \
  --label_len $seq_len \
  --random_seed $random_seed \
  --max_grad_norm $max_grad_norm \
  --scale $scale \
  --train_epochs 1000 \
  --itr 1 \
  --batch_size $batch_size \
  --batch_size_test $batch_size_test \
  --plot_every $plot_every \
  --learning_rate $learning_rate \
  --transfer_learning $transfer_learning \
  --fix_embedding $fix_embedding \
  --gradient_accumulation $gradient_accumulation\
  --resume $resume >logs/LongForecasting/$model_name'_'$data_alias'_'$random_seed'_RD('$random_drop_training')_'$seq_len'_'$lookback'_'$pred_len'_K('$e_layers')_d('$d_model')_Dropout('$dropout')_m('$sampling_step')_W('$linear_warmup_steps')_Limit('$token_limit')_wRet('$token_retriever_flag')'
  
done