kernel=$1
dataset=$2

python ./main.py   \
    --data_source libsvm \
    --dataset_name ${dataset} \
    --base base \
    --method svc \
    --module resnet18 \
    --batch_size 1024 \
    --num_workers 4 \
    --max_evals 10 \
    --kernel ${kernel} \
    --C 1.0 \
    --coef0 0.0 \
    --degree 3 \
    --prototype_generation_k_fold 3 \
    --api_keys "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMWRkNzhhZi04NTYzLTRkNTktYWM1Yi1iODdlZWI2NTNmMTMifQ==" \
    --logger_project_name "cfnp/svc" \
    --logger_run_name base_svc_rbf_ratio=0.9 \
    --logger_description "run pipelines, base model=base, method=svc, kernel=${kernel}, default" \
    --max_epochs 30 \
    --num_sanity_val_steps 0 \
    --optimizer adam \
    --lr 0.1 \
    --gpus 1 \
    --run_baselines \
    --checkpoint_save_last