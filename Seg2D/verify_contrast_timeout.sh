#!/bin/bash 

# what are the variables for this analysis?

N=10 # NUMBER OF IMAGES

# echo $N 

# Get image indexes from this range
IDXS=$(shuf -i 101-415 -n $N --random-source=<(yes 2024))

# echo $IDXS

# sliceSize="64 80 96"
# gamma="0.5 1 2" # lower and upper bound for typical ranges used for gamma
# gamma_range="0.005 0.0075 0.01" # gamma ranges to consider for each gamma value
sliceSize="64"
gamma="0.5"
gamma_range="0.005"

reachMethod="relax-star-range"
relaxFactor="0.95"

# echo $relaxFactor

perturbation="AdjustContrast"

# Generate combos to then execute in parallel
declare -A combos=()
count=$1
for i in $sliceSize
do
    for j in $gamma
    do
        for k in $gamma_range
        do
            for img in $IDXS
            do

            # (killall MathWorksServiceHost; # ensure none are running
            (sleep 10; # wait a second to ensure matlab is closed and ready to run again
            # adding an & at the end would execute in parallel, but matlab fails way too often like this
            # adding the pause seems to reduce a lot of the errors (at least the undetected toolboxes)
            timeout 450 matlab -nodesktop -nojvm -nosplash -minimize -noFigureWindows -r "pause(5); addpath(genpath('"../../nnv/code/nnv"')); verify_model_monai_image $i $j $k $img $reachMethod $relaxFactor $perturbation; quit") &
            # killall MathWorksServiceHost) & # close all matlab in case it did not exit properly

            done
        done
    done
done

exit 0