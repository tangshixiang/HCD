python -u examples/online_clustering_uda_label_propagation_final.py \
    --logs-dir demo --merge-alpha 0.99 --sp-method 12 --thre 0.25 0.25 0.25 --iterative 2 --update-method 4 \
    --method 17 --neighbor-num 64 --split-num 8 --topk-num 0.45 --split-anchor-thre 1.0 --k1 20 --merge-wo-outlier 0 \
    --connect-num 30 --split-alpha 0.99 --split-gap 2 --update-before 0 -dt market1501 -ds dukemtmc --iters 400
