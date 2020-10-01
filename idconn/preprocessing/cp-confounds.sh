while [[ $# -gt 0 ]]; do
		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-session-1_retr-0_confounds.txt /home/data/nbc/physics-learning/retrieval-graphtheory/output/post/$1/$1-session-1_retr-0_confounds.txt

		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-session-1_retr-1_confounds.txt /home/data/nbc/physics-learning/retrieval-graphtheory/output/post/$1/$1-session-1_retr-1_confounds.txt

		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-session-0_retr-0_confounds.txt /home/data/nbc/physics-learning/retrieval-graphtheory/output/pre/$1/$1-session-0_retr-0_confounds.txt

		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-session-0_retr-1_confounds.txt /home/data/nbc/physics-learning/retrieval-graphtheory/output/pre/$1/$1-session-0_retr-1_confounds.txt

shift
done
