# bash/train.sh -d amazon -e Grocery_2014 -f remove_100bigger_thres5 -s 42 -c 1 -b grocery_test -a NeuMF

while getopts ':d:e:s:c:b:a:f:m:' opt
do
    case $opt in
        d)
        dataset1="$OPTARG" ;;
        e)
        dataset2="$OPTARG" ;;
        f)
        dataset3="$OPTARG" ;;
        s)
        seed="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        a)
        arch="$OPTARG" ;;
        m)
        mode="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done


gradient_clip_val=1
warmup_ratio=0.1
weight_decay=1e-5

precision=16
batch_size=65536
eval_batch_size=64000
learning_rate=50
max_epochs=100

l2=500

######################### TODO #########################
# designate the directory
data_dir=""
output_dir=""
########################################################

CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train.py \
  --gpus=1 \
  --dataset1=${dataset1} \
  --dataset2=${dataset2} \
  --dataset3=${dataset3} \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --output_dir "${output_dir}" \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --seed $seed \
  --warmup_ratio ${warmup_ratio} \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_epochs ${max_epochs} \
  --output_sub_dir ${subname} \
  --arch ${arch} \
  --l2 ${l2} \
  --mode ${mode} 