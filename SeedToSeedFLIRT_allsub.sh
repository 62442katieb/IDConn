while [[ $# -gt 0 ]]; do
	mkdir /home/data/nbc/physics-learning/data/first-level/$1/session-1/retr/mni/

	applywarp --ref=/home/applications/fsl/5.0.8/data/standard/MNI152_T1_2mm_brain.nii.gz --in=/home/data/nbc/physics-learning/data/first-level/$1/session-1/retr/retr-0/retr-5mm.feat/filtered_func_data.nii.gz --out=/home/data/nbc/physics-learning/data/first-level/$1/session-1/retr/mni/$1_filtered_func_data_0.nii.gz --warp=/home/data/nbc/physics-learning/data/first-level/$1/session-1/retr/retr-0/retr-5mm.feat/reg/example_func2standard_warp.nii.gz
	applywarp --ref=/home/applications/fsl/5.0.8/data/standard/MNI152_T1_2mm_brain.nii.gz --in=/home/data/nbc/physics-learning/data/first-level/$1/session-1/retr/retr-1/retr-5mm.feat/filtered_func_data.nii.gz --out=/home/data/nbc/physics-learning/data/first-level/$1/session-1/retr/mni/$1_filtered_func_data_1.nii.gz --warp=/home/data/nbc/physics-learning/data/first-level/$1/session-1/retr/retr-1/retr-5mm.feat/reg/example_func2standard_warp.nii.gz

	#cp /home/data/nbc/SeedToSeed/$1_filtered_func_data_mni.nii.gz /home/data/nbc/anxiety-physics/pre/$1_filtered_func_data_mni.nii.gz

shift
done
