Anaconda_path="XXX" # please replace XXX with the real path to the newly installed Anaconda

if [ ${Anaconda_path} = "XXX" ]; then
    echo "Please replace XXX with the real path to the newly installed Anaconda."
    exit
fi

echo "**********************************************"
echo "**********************************************"
echo "Install Python package mrcfile"
${Anaconda_path}/bin/pip install mrcfile
echo ""

echo "Install Python package scikit-image"
${Anaconda_path}/bin/pip install scikit-image
echo ""

echo "**********************************************"
echo "**********************************************"
echo "Compile QRNAS software for energy minimization"
cd ${HOME}/DeepCryoRNA/src/QRNAS
make
echo ""

echo "**********************************************"
echo "**********************************************"
cd ${HOME}/DeepCryoRNA/src/
g++ -O3 -shared -fPIC alignment_score.cpp -o alignment_score.so -std=c++11
echo ""