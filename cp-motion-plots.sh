while [[ $# -gt 0 ]]; do
		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-0_retr-metrics.png /home/data/nbc/physics-learning/retrieval-graphtheory/output/qa/$1-0_retr-metrics.png
		cp /home/data/nbc/physics-learning/retrieval-graphtheory/output/$1/$1-1_retr-metrics.png /home/data/nbc/physics-learning/retrieval-graphtheory/output/qa/$1-1_retr-metrics.png

shift
done
