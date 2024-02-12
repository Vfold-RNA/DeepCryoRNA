Anaconda_path=$1

echo "Anaconda_path: $Anaconda_path"
if ! [ -f ${Anaconda_path}/bin/pip ]; then
    echo "The Anaconda_path does not exist!"
    exit
fi

echo "**********************************************"
echo "**********************************************"
echo "Install Python package mrcfile"
${Anaconda_path}/bin/pip install mrcfile
echo ""

echo "Install Python package scikit-image"
${Anaconda_path}/bin/pip install scikit-image==0.20.0
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
