#include <iostream>
#include<opencv2/opencv.hpp>	
#include"ASW.h"
using namespace cv;
using namespace std;

int main()
{
    Mat left = imread("im2.png");
	const Mat right = imread("im6.png");
	
	// 图片尺寸，一般来说要判断两张输入的照片尺寸是否一致
	int row = left.rows;
	int col = left.cols;

	// 创建存储空间
	Mat leftLab = Mat::zeros(row, col, CV_32FC3);
	Mat rightLab = Mat::zeros(row, col, CV_32FC3);

	Mat leftGray = Mat::zeros(row, col, CV_32FC1);
	Mat rightGray = Mat::zeros(row, col, CV_32FC1);

	Mat colorWeight = Mat::zeros(row, col, CV_32FC1);
	Mat leftPaded, leftPadLab;
	Mat rightPaded, rightPadLab;

	Mat dispLeft = Mat::zeros(row, col, CV_32FC1);
	Mat dispRight = Mat::zeros(row, col, CV_32FC1);
	Mat lastDisp = Mat::zeros(row, col, CV_8UC1);

	// cvt to LAB
	cvtColor(left, leftLab, CV_BGR2Lab);
	cvtColor(right, rightLab, CV_BGR2Lab);
	// cvt to gray
	cvtColor(left, leftGray, CV_BGR2GRAY);
	cvtColor(right, rightGray, CV_BGR2GRAY);

	// innitial 
	int winSize = 11;  // 窗口实际大小为 2*winsize+3
	int dispRange = 60;
	int spaceSigma = 50;
	int colorSigma = 25;
	cv::Mat spaceMask;
	std::vector<double> colorMask;
	getGausssianMask(spaceMask, cv::Size(winSize * 2 + 3, winSize * 2 + 3), spaceSigma);//空间模板
	getColorMask(colorMask, colorSigma);//值域模板

	// 扩充边界 
	Mat target;

	//bilateralFilter()
	bilateralfiter(left, target, winSize, spaceMask, colorMask);
}