run_list='candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'

for r in $run_list
do
    accelerate launch Reversedistillation.py --yaml_config ./configs/visa/visa_$r.yaml
done 

mv_list='bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'
for r in $mv_list
do
    accelerate launch Reversedistillation.py --yaml_config ./configs/mvtec/mvtec_$r.yaml
done 