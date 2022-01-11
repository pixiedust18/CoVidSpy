# CoVidSpy - A Pandemic Management and Safety Norm Violation Detection System
This is a computer vision based project that tracks people and detects whether or not they're wearing masks, and maintaining the recommended social distance at an extremely high FPS for real time usage. Fever detection has also been included using temperature estimation.

Result videos and pictures for :
- Mask detection
- Social distancing ( With zoning )
- Social distancing + Mask Detection 
- Human temperature estimation

https://drive.google.com/drive/folders/1ljjTYc5eigNaVWz9o84qDD3Vh7eWZW1F?usp=sharing


**To run social distancing combined with mask detection, run command :** 
- **./run.sh**

**To run zoning, with mask and social distancing follow the given steps :**
1) Access the notebook Zoning.ipynb 
  - Enter the number of zones the floor area will be divided into
  - Enter the 4 coordinates of the floor area(p1, p2, p3, p4), so that lines p1-p4 and p2-p3 are perpendicular to the zoning division lines, and run Zoning.ipynb
2) Access the folder - "floor_coordinates.txt"
  - Copy the output to "floor_coordinates.txt"
3) Finally, run the file run_zoning.py

Find a detailed explanation and demo of the project here: <br>
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/53fDd14n36Q/0.jpg)](https://www.youtube.com/watch?v=53fDd14n36Q)

If you have used this project in your research, do cite our paper(https://ieeexplore.ieee.org/document/9297902): <br>
K. Bhambani, T. Jain and K. A. Sultanpure, "Real-time Face Mask and Social Distancing Violation Detection System using YOLO," 2020 IEEE Bangalore Humanitarian Technology Conference (B-HTC), 2020, pp. 1-6, doi: 10.1109/B-HTC50970.2020.9297902.

