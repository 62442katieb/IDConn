while [[ $# -gt 0 ]]; do
		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-gen-corrmat.csv /home/kbott006/physics-retrieval/output/$1-gen-corrmat.csv
		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-phy-corrmat.csv /home/kbott006/physics-retrieval/output/$1-phy-corrmat.csv

shift
done
