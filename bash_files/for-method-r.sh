kernel=$1
method=$2

echo $kernel >> logs/${method}/${kernel}_log.txt
for dataset in cadata housing_scale abalone_scale cpusmall_scale
do
    echo $dataset >> logs/${method}/${kernel}_log.txt
    # run limited memory default
    echo run limited memory default >> logs/${method}/${kernel}_log.txt
    bash bash_files/${method}/${method}_default.sh $kernel $dataset
    
    # run cmp_ratio=0.9
    echo run cmp_ratio=0.9 >> logs/${method}/${kernel}_log.txt
    bash bash_files/${method}/${method}_cmp_ratio ${kernel} ${dataset} 0.9

done