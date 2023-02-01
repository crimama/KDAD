mv_list='candle'
for r in $mv_list
do
    accelerate launch Reversedistillation.py --yaml_config ./configs/visa/visa_$r.yaml
done 
#edit 