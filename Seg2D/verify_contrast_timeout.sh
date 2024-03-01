#!/bin/bash 

# what are the variables for this analysis?

N=20 # NUMBER OF IMAGES

# Get image indexes from this range
IDXS=$(shuf -i 101-415 -n $N --random-source=<(yes 2024))

sliceSize="64 80 96"
gamma="0.5 1 2" # lower and upper bound for typical ranges used for gamma
gamma_range="0.005 0.0075 0.01" # gamma ranges to consider for each gamma value
# sliceSize="64"
# gamma="0.5"
# gamma_range="0.005"

reachMethod="relax-star-range"
relaxFactor="0.95"


perturbation="AdjustContrast"

for i in $sliceSize
do
    for j in $gamma
    do
        for k in $gamma_range
        do
            for img in $IDXS
            do

            echo $i $j $k $img

            # adding an & at the end would execute in parallel, but matlab fails way too often like this
            # adding the pause seems to reduce a lot of the errors (at least the undetected toolboxes), but not eliminate them completely
            timeout 450 matlab -r "addpath(genpath('"../../nnv/code/nnv"')); verify_model_monai_image $i $img $reachMethod $relaxFactor $perturbation $j $k; pause(1); quit"

            killall -q matlab

            echo "Done. Closing MATLAB now..."

            done
        done
    done
done