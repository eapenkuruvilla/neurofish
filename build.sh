python setup.py build_ext --inplace
if [ $? -eq 0 ]; then
    rm nn_ops_fast.c
    mkdir -p libs
    echo "Moving *.so to the libs directory"
    mv *.so ./libs/
fi



