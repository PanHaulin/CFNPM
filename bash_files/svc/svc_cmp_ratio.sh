kernel=$1
dataset=$2
cmp_ratio=$3

python ./main.py   \
    --data_source libsvm \
    --dataset_name ${dataset} \
    --base base \
    --method svc \
    --module resnet18 \
    --batch_size 256 \
    --num_workers 24 \
    --max_evals 1 \
    --kernel ${kernel} \
    --C 1.0 \
    --coef0 0.0 \
    --degree 3 \
    --api_keys "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMWRkNzhhZi04NTYzLTRkNTktYWM1Yi1iODdlZWI2NTNmMTMifQ==" \
    --logger_project_name "cfnp/test" \
    --logger_run_name "test: cfnp-d setting" \
    --logger_description "cfnp-d setting" \
    --max_epochs 250 \
    --num_sanity_val_steps 0 \
    --optimizer adam \
    --lr 0.01 \
    --gpus 1 \
    --checkpoint_save_last \
    --cmp_ratio ${cmp_ratio}
    # --run_baselines \
    # --prototype_generation_k_fold 3 \