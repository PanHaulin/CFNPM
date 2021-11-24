python ./main.py   \
    --data_source libsvm \
    --dataset_name a9a \
    --base base \
    --method svc \
    --module resnet18 \
    --batch_size 1024 \
    --num_workers 8 \
    --cmp_ratio 0.9 \
    --max_evals 1 \
    --kernel linear \
    --C 1.0 \
    --coef0 0.0 \
    --degree 3 \
    --prototype_generation_k_fold 3 \
    --api_keys "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMWRkNzhhZi04NTYzLTRkNTktYWM1Yi1iODdlZWI2NTNmMTMifQ==" \
    --logger_project_name "cfnp/svc" \
    --logger_run_name base_svc_rbf_ratio=0.9 \
    --logger_description "run pipelines, base model=base, method=svc, kernel=linear, cmp_ratio=0.9" \
    --patience 10 \
    --max_epochs 100 \
    --num_sanity_val_steps 0 \
    --optimizer adam \
    --lr 0.1 \
    --gpus 1 \
    --run_baselines \
    # --fast_dev_run \
    # --prototype_generation_k_fold 1 \
    # --prototype_generation_sampling_strategy maintain \
    # --prototype_selection_sampling_strategy maintain \
