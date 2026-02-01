python setup.py build_ext --inplace
if [ $? -eq 0 ]; then
    rm nn_ops_fast.c
fi

echo "Moving *.so to the libs directory"
mv *.so ./libs/

