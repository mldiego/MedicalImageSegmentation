#!/bin/bash 

# what are the variables for this analysis?

N=20 # NUMBER OF IMAGES

# Get image indexes from this range
IDXS=$(shuf -i 101-415 -n $N --random-source=<(yes 2024))

sliceSize="64 80 96"
order="3"
coeff="0.1 0.25 0.5"
coeff_range="0.00025 0.0005 0.001"

reachMethod="relax-star-range"
relaxFactor="0.95"


perturbation="BiasField"

for i in $sliceSize
do
    for j in $coeff
    do
        for k in $coeff_range
        do
            for img in $IDXS
            do

            echo $i $j $k $img

            # adding an & at the end would execute in parallel, but matlab fails way too often like this
            # adding the pause seems to reduce a lot of the errors (at least the undetected toolboxes), but not eliminate them completely
            timeout 450 matlab -r "addpath(genpath('"../../nnv/code/nnv"')); verify_model_monai_image $i $img $reachMethod $relaxFactor $perturbation $order $j $k; pause(1); quit"

            killall -q matlab

            echo "Done. Closing MATLAB now..."

            done
        done
    done
done