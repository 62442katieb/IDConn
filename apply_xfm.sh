while [[ $# -gt 0 ]]; do

    applywarp -i /home/kbott006/physics-retrieval/idconn-retrieval/18-networks-5.14-mni_2mm_regions.nii.gz -r /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-0_retr-example_func.nii.gz -o /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-0_18_icn-regions_retr.nii.gz -w /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-0_mni-fnirt-epi-warp.nii.gz --interp=nn

    applywarp -i /home/kbott006/physics-retrieval/idconn-retrieval/18-networks-5.14-mni_2mm_regions.nii.gz -r /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-1_retr-example_func.nii.gz -o /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-1_18_icn-regions_retr.nii.gz -w /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-1_mni-fnirt-epi-warp.nii.gz --interp=nn


shift
done
