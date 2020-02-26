while [[ $# -gt 0 ]]; do
		cp /home/data/nbc/physics-learning/data/pre-processed/$1/session-0/fci/fci-0/fci-0-ppi.feat/mc/prefiltered_func_data_mcf.par /home/data/nbc/physics-learning/retrieval-graphtheory/output/pre/$1/$1-session-0_fci-0_mcf.nii.gz.par

		cp /home/data/nbc/physics-learning/data/pre-processed/$1/session-0/fci/fci-1/fci-1-ppi.feat/mc/prefiltered_func_data_mcf.par /home/data/nbc/physics-learning/retrieval-graphtheory/output/pre/$1/$1-session-0_fci-1_mcf.nii.gz.par

		cp /home/data/nbc/physics-learning/data/pre-processed/$1/session-0/fci/fci-2/fci-2-ppi.feat/mc/prefiltered_func_data_mcf.par /home/data/nbc/physics-learning/retrieval-graphtheory/output/pre/$1/$1-session-0_fci-2_mcf.nii.gz.par

		cp /home/data/nbc/physics-learning/data/pre-processed/$1/session-1/fci/fci-0/fci-0-ppi.feat/mc/prefiltered_func_data_mcf.par /home/data/nbc/physics-learning/retrieval-graphtheory/output/post/$1/$1-session-1_fci-0_mcf.nii.gz.par

		cp /home/data/nbc/physics-learning/data/pre-processed/$1/session-1/fci/fci-1/fci-1-ppi.feat/mc/prefiltered_func_data_mcf.par /home/data/nbc/physics-learning/retrieval-graphtheory/output/post/$1/$1-session-1_fci-1_mcf.nii.gz.par

		cp /home/data/nbc/physics-learning/data/pre-processed/$1/session-1/fci/fci-2/fci-2-ppi.feat/mc/prefiltered_func_data_mcf.par /home/data/nbc/physics-learning/retrieval-graphtheory/output/post/$1/$1-session-1_fci-2_mcf.nii.gz.par


shift
done
