OUTPUT=SMGF.log
{
    python SMGF.py --dataset acm --knn_k 10
    python SMGF.py --dataset dblp --knn_k 10
    python SMGF.py --dataset imdb --knn_k 500
    python SMGF.py --dataset yelp --knn_k 200
    python SMGF.py --dataset freebase --knn_k 10
    python SMGF.py --dataset rm --knn_k 10
    python SMGF.py --dataset amazon-photos --knn_k 10
    python SMGF.py --dataset amazon-computers --knn_k 10
    python SMGF.py --dataset mageng --knn_k 10
    python SMGF.py --dataset magphy --knn_k 10
}|tee -a $OUTPUT

OUTPUT1=SMGF_OPT.log
{
    python SMGF_OPT.py --dataset acm --knn_k 10
    python SMGF_OPT.py --dataset dblp --knn_k 10
    python SMGF_OPT.py --dataset imdb --knn_k 500
    python SMGF_OPT.py --dataset yelp --knn_k 200
    python SMGF_OPT.py --dataset freebase --knn_k 10
    python SMGF_OPT.py --dataset rm --knn_k 10
    python SMGF_OPT.py --dataset amazon-photos --knn_k 10
    python SMGF_OPT.py --dataset amazon-computers --knn_k 10
    python SMGF_OPT.py --dataset mageng --knn_k 10
    python SMGF_OPT.py --dataset magphy --knn_k 10
}|tee -a $OUTPUT1
