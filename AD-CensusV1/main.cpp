#include <iostream>
#include <vector>	
#include <opencv2/opencv.hpp>	
#include "AD-Census.h"	
#include "CrossArm.h"
#include "ScanlineOptimizer.h"
#include "PostProcessing.h"	

using namespace std;
using namespace cv;


int main()
{
	// 读取图像
	auto leftImage = imread("im2.png");
	auto rightImage = imread("im6.png");
	Mat leftGray, rightGray;
	cvtColor(leftImage, leftGray, CV_BGR2GRAY);
	cvtColor(rightImage, rightGray, CV_BGR2GRAY);

	int row = leftImage.rows;
	int col = leftImage.cols;
	int dispRange = 60;
	float sigmaS = 30;
	float sigmaC = 10;
	int tao = 30;
	int p1 = 10;
	int p2 = 150;
	int gate = 2;

	//float* leftptr, * rightptr, *leftDisp;
	float* leftptr = new float[row * col]();
	float* rightptr = new float[row * col]();
	float* leftDisp = new float[row * col]();
	float* rightDisp = new float[row * col]();
	float* lastDisp = new float[row * col]();
	float* aggredCostVolumeLeft = new float[row * col * dispRange]();
	float* aggredCostVolumeRight = new float[row * col * dispRange]();
	// 遮挡区像素集
	std::vector<std::pair<int, int>> occlusions;
	// 误匹配区像素集
	std::vector<std::pair<int, int>> mismatches;
	//float* costVolumePtr = new float[row * col * dispRange]();

	for (int i = 0; i < row; i++)
	{
		for(int j =0; j<col; j++)
		{
			auto left = leftGray.at<uchar>(i, j);
			auto right = rightGray.at<uchar>(i, j);
			leftptr[i * col + j] = static_cast<float>(left);
			rightptr[i * col + j] = static_cast<float>(right);
		}
	}
	// 程序初始化 
	AD_Census ADcensus;
	ADcensus.Initialize(leftptr, rightptr, dispRange, row, col, leftGray, rightGray, sigmaC, sigmaS);   
	ADcensus.ComputeADcensus();
	ADcensus.ComputeADcensusRight();
	ADcensus.WTA(leftDisp, rightDisp);
	float* costVolumeLeftPtr = ADcensus.GetPtrLeft();
	float* costVolumeRightPtr = ADcensus.GetPtrRight();


	// 代价聚合
	CrossArmAggregation CrossArm;
	CrossArm.Initialize(row, col, leftptr, rightptr, tao, dispRange);
	CrossArm.ComputeLeftArmLength(leftGray);
	CrossArm.ComputeRightArmLength(leftGray);
	CrossArm.ComputeTopArmLength(leftGray);
	CrossArm.ComputeButtonArmLength(leftGray);
	//CrossArm.Aggregation(costVolumePtr, aggredCostVolume); // 水平聚合
	CrossArm.AggregationVertical(costVolumeLeftPtr, aggredCostVolumeLeft); // 垂直聚合
	CrossArm.WTA(aggredCostVolumeLeft, leftDisp);
	// 右聚合
	CrossArm.Initialize(row, col, leftptr, rightptr, tao, dispRange);
	CrossArm.ComputeLeftArmLength(rightGray);
	CrossArm.ComputeRightArmLength(rightGray);
	CrossArm.ComputeTopArmLength(rightGray);
	CrossArm.ComputeButtonArmLength(rightGray);
	//CrossArm.Aggregation(costVolumePtr, aggredCostVolume); // 水平聚合
	CrossArm.AggregationVertical(costVolumeRightPtr, aggredCostVolumeRight); // 垂直聚合
	CrossArm.WTA(aggredCostVolumeRight, rightDisp);
	// 线扫描优化
	//ScanlineOptimizer ScanlineOpt;
	//ScanlineOpt.Initialize(row, col, dispRange, aggredCostVolume, p1, p2);
	//ScanlineOpt.ScanLine(aggredCostVolume, leftptr);
	//ScanlineOpt.WTA(leftDisp);

	// 后处理
	//LeftRightConsistency(col, row, gate, leftDisp, rightDisp, occlusions, mismatches);
	//RemoveSpeckles(leftDisp, col, row, 1, 30, Invalid_Float);
	//MedianFilter(leftDisp, lastDisp, col, row, 3);


	// 格式转换
	Mat leftDispShow =Mat(row, col, CV_8UC1);
	Mat rightDispShow = Mat(row, col, CV_8UC1);
	Mat lastDispShow = Mat(row, col, CV_8UC1);
	TransformToShow(leftDisp, leftDispShow, row, col);
	TransformToShow(rightDisp, rightDispShow, row, col);
	TransformToShow(lastDisp, lastDispShow, row, col);

	// 标准化
	normalize(leftDispShow, leftDispShow, 0, 255, NORM_MINMAX);
	leftDispShow.convertTo(leftDispShow, CV_8UC1);
	normalize(rightDispShow, rightDispShow, 0, 255, NORM_MINMAX);
	rightDispShow.convertTo(rightDispShow, CV_8UC1);
	normalize(lastDispShow, lastDispShow, 0, 255, NORM_MINMAX);
	lastDispShow.convertTo(lastDispShow, CV_8UC1);
	imshow("rightDisp", rightDispShow);
	imshow("leftDisp", leftDispShow);
	imshow("lastDisp", lastDispShow);
	imwrite("lastDisp.png", lastDispShow);
	imwrite("rightDisp.png", rightDispShow);
	imwrite("leftDisp.png", leftDispShow);
	waitKey();

	return 0;
}