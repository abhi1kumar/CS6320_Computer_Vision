echo -e "\n--------------------------------------------------"
echo -e "\tQ1: Belief Propagation\t"
echo -e "--------------------------------------------------"
python3 Functions/q1.py 

echo -e "\n--------------------------------------------------"
echo -e "\tQ2: Stereo Matching using BP Algorithm\t"
echo -e "--------------------------------------------------"
python3 Functions/q2.py -l Inputs/left1.png -r Inputs/right1.png -w 1
python3 Functions/q2.py -l Inputs/left1.png -r Inputs/right1.png -w 10
python3 Functions/q2.py -l Inputs/left1.png -r Inputs/right1.png -w 50
python3 Functions/q2.py -l Inputs/left1.png -r Inputs/right1.png -w 100

python3 Functions/q2.py -l Inputs/left2.png -r Inputs/right2.png -w 1
python3 Functions/q2.py -l Inputs/left2.png -r Inputs/right2.png -w 10
python3 Functions/q2.py -l Inputs/left2.png -r Inputs/right2.png -w 50
python3 Functions/q2.py -l Inputs/left2.png -r Inputs/right2.png -w 100

#python3 Functions/q3.py -l Inputs/left3.bmp -r Inputs/right3.bmp
