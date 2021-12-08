#include<iostream>	
#include<opencv2/opencv.hpp>	
#include"NCC.h"

using namespace std;
using namespace cv;

int main()
{
	// 读取图像
	auto leftImage = imread("left8.png", 0);
    auto rightImage = imread("right8.png", 0);

	// 初始化变量
	int col = leftImage.cols;
	int row = leftImage.rows;
	int winSize = 10;
	int dispRange = 200;

	Mat leftDisp = Mat::zeros(row,col, CV_32S);
	Mat rightDisp = Mat::zeros(row, col, CV_32S);
	

	//auto left_disp = ncc(leftImage, rightImage, "left");
	//cout << "finished !" << endl;

	//imshow("disp", left_disp);
	//imwrite("disp_left.png",left_disp);
	//waitKey();

	
	// 获取视差
	NCC_algorithem(leftImage, rightImage, col,row,leftDisp, winSize, dispRange);
	/*for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
			cout << "left disp: " << leftDisp.at<int>(i, j) << endl;

	}*/
	//imshow("left", leftImage);
	//imshow("right", rightImage);
	//waitKey();
	 //视差转换显示
	normalize(leftDisp, leftDisp, 255, 0, NORM_MINMAX);
	leftDisp.convertTo(leftDisp, CV_8UC1);
	///*for (auto i = 0; i < row; i++)
	//{
	//	for (auto j = 0; j < col; j++)
	//		cout << "left disp: " << leftDisp.at<int>(i,j) << endl;
	//	
	//}*/
	////imshow("left_disp", leftDisp);
	cout << "finished" << endl;
	//leftDisp = leftDisp + 10;
	imshow("leftdisp", leftDisp);
	imwrite("left__disp.png", leftDisp);
	waitKey();

	return 0;
}