while [[ $# -gt 0 ]]; do
    fslmaths roi$1.nii.gz -mul $1 roi$1-$1.nii.gz
shift
done