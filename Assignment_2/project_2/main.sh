echo -e "\n--------------------------------------------------"
echo -e "\tQ1: Line Detection\t"
echo -e "--------------------------------------------------"
python3 Functions/q1.py -i Inputs/lineDetect1.bmp
python3 Functions/q1.py -i Inputs/lineDetect2.bmp
python3 Functions/q1.py -i Inputs/lineDetect3.bmp
python3 Functions/q1.py -i Inputs/img1.png

echo -e "\n--------------------------------------------------"
echo -e "\tQ2: Sky Detection\t"
echo -e "--------------------------------------------------"
python3 Functions/q2.py -i Inputs/detectSky1.bmp
python3 Functions/q2.py -i Inputs/detectSky2.bmp
python3 Functions/q2.py -i Inputs/detectSky3.bmp

echo -e "\n--------------------------------------------------"
echo -e "\tQ3: Bag of Words and Vocabulary Tree\t"
echo -e "--------------------------------------------------"
python3 Functions/q3.py -l Inputs/left1.png -r Inputs/right1.png
python3 Functions/q3.py -l Inputs/left2.png -r Inputs/right2.png
python3 Functions/q3.py -l Inputs/left3.bmp -r Inputs/right3.bmp
