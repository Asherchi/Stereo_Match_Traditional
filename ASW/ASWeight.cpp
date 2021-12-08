#include <iostream>
#include<opencv2/opencv.hpp>	
#include"ASW.h"
using namespace cv;
using namespace std;

int main()
{
	// 读取图片
	const Mat left = imread("im2.png");
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

	imwrite("leftgray.png", leftGray);
	imwrite("rightgray.png", rightGray);

	// innitial 
	int winSize = 11;  // 窗口实际大小为 2*winsize+3  16 50 20 40
	int dispRange = 60;
	int spaceSigma = 50;
	int colorSigma = 30;
	int T = 40;
	cv::Mat spaceMask;
	std::vector<double> colorMask;
	getGausssianMask(spaceMask, cv::Size(winSize*2+3,winSize*2+3), spaceSigma);//空间模板
	getColorMask(colorMask, colorSigma);//值域模板

	// 扩充边界 
	copyMakeBorder(leftGray, leftPaded, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
	copyMakeBorder(rightGray, rightPaded, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
	copyMakeBorder(leftLab, leftPadLab, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
	copyMakeBorder(rightLab, rightPadLab, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);

	// process data
	AdaptiveSupportWeight(dispLeft, leftPaded, rightPaded, winSize, dispRange, leftPadLab, rightPadLab, spaceMask, colorMask,T = 40);
	AdaptiveSupportWeightRight(dispRight, leftPaded, rightPaded, winSize, dispRange, leftPadLab, rightPadLab, spaceMask, colorMask, T =40);
	//normalize(disp, disp, 0, 255, NORM_MINMAX);
	//disp.convertTo(disp, CV_8UC1);
	//cv::filterSpeckles(disp, 0, 20, 1);
	//medianBlur(disp, disp, 5);
	CrossCheckDiaparity(dispLeft, dispRight, lastDisp, dispRange, 3);
	normalize(dispLeft, dispLeft, 0, 255, NORM_MINMAX);
	normalize(dispRight, dispRight, 0, 255, NORM_MINMAX);
	normalize(lastDisp, lastDisp, 0, 255, NORM_MINMAX);
	dispLeft.convertTo(dispLeft, CV_8UC1);
	dispRight.convertTo(dispRight, CV_8UC1);
	lastDisp.convertTo(lastDisp, CV_8UC1);
	filterSpeckles(lastDisp, 0, 40, 2);
	medianBlur(lastDisp, lastDisp, 5);
	imwrite("speckle.png", lastDisp);
	FillImageNew(lastDisp);
	imwrite("filled.png", lastDisp);
	medianBlur(lastDisp, lastDisp, 3);
	imwrite("medain.png", lastDisp);
	//FillImageLast(lastDisp);

	//normalize(dispLeft, dispLeft, 0, 255, NORM_MINMAX);
	//normalize(dispRight, dispRight, 0, 255, NORM_MINMAX);
	//normalize(lastDisp, lastDisp, 0, 255, NORM_MINMAX);
	//dispLeft.convertTo(dispLeft, CV_8UC1);
	//dispRight.convertTo(dispRight, CV_8UC1);
	//lastDisp.convertTo(lastDisp, CV_8UC1);
	imshow("lastDisp", lastDisp);
	imshow("displeft", dispLeft);
	imshow("dispRight", dispRight);
	imwrite("displeft.png", dispLeft);
	imwrite("dispRight.png", dispRight);
	imwrite("lastDisp.png", lastDisp);
	waitKey();


	return 0;
}