#!/bin/bash
N=$1

datalist=("1_ALOI" "3_backdoor" "4_breastw" "5_campaign" "6_cardio" "9_census" "12_fault" "15_Hepatitis" "17_InternetAds" "19_landsat" "20_letter" "24_mnist" "25_musk" "32_shuttle" "36_speech") #
n_list=("5" "5" "4" "5" "4" "5" "4" "2" "4" "4" "4" "4" "4" "5" "4")

for i in "${!datalist[@]}"
  do
    data=${datalist[$i]}
    n=${n_list[$i]}
    echo $data
    for seed in 0 1 2 3 49
        do
            python3.13 CLOE/CLOE/main.py --data-name "$data" --seed "$seed" --n "$n"
            python3.13 CLOE/CLOE/threshold_study.py --data-name "$data" --seed "$seed"
            python3.13 CLOE/CLOE/ablation.py --data-name "$data" --study 0 --n "$n"  --seed "$seed"
            python3.13 CLOE/CLOE/ablation.py --data-name "$data" --study 1 --n "$n" --nb-epochs 150  --seed "$seed"
            python3.13 CLOE/CLOE/ablation.py --data-name "$data" --study 2 --n "$n"  --seed "$seed"
            python3.13 CLOE/CLOE/ablation.py --data-name "$data" --study 3 --n "$n"  --seed "$seed"
            python3.13 CLOE/baseline/train_test.py --data-name "$data" --seed "$seed" --oc-svm True --iforest True --knn True --kde True --ecod True --deepSVDD True 
            python3.13 CLOE/baseline/DRL/main.py --dataname "$data" --model_type DRL --preprocess standard --diversity True --plearn False --input_info True --input_info_ratio 0.1 --cl True --cl_ratio 0.06 --basis_vector_num 5 --seed "$seed"
            python3.13 CLOE/baseline/RCA/trainRCA.py --data "$data" --seed "$seed" --training_ratio 0.599 --max_epochs 200 --hidden_dim 128 --z_dim 10
            python3.13 CLOE/baseline/MCM/main.py --data-name "$data" --seed "$seed"
            python3.13 CLOE/baseline/DDAE/run.py --data-name "$data" --seed "$seed"
            if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
                # wait only for first job
                wait -n
            fi
        done
  done

 python3.13 CLOE/results/test.py