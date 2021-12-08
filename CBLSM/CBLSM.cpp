#include <iostream>	
#include <vector>
#include <opencv2/opencv.hpp>	
#include "CBLSM.h"
#include "cross_aggregator.h"	
#include "adcensus_types.h"
#include "PostProcessing.h"


using namespace cv;
using namespace std;

int main()
{
	// read Image
    auto leftColor = imread("im2.png");
	auto rightColor = imread("im6.png");

	// initial
	Mat imageL, imageR;
	cvtColor(leftColor, imageL, CV_BGR2GRAY);
	cvtColor(rightColor, imageR, CV_BGR2GRAY);
	Mat armImageL, armImageR;
	medianBlur(imageL, armImageL, 3);
	medianBlur(imageR, armImageR, 3);
	auto col = imageL.cols;
	auto row = imageL.rows;
	int winSize = 1;
	int dispRange = 60;
	uchar tao = 25;
	int maxLength = 34;
	int secLength = 17;
	auto dispVolumLeft = new float[row* col * dispRange]();  // �������ۿռ�
	auto dispVolumRight = new float[row * col * dispRange]();  // �������ۿռ�
	auto costVolumLeft = new float[row * col * dispRange]();
	auto costVolumRight = new float[row * col * dispRange]();
	auto costVolumLeftSec = new float[row * col * dispRange]();
	auto costVolumRightSec = new float[row * col * dispRange]();
	ADCensusOption option;

	// ����洢��۱۳�
	auto ArmLL = new int[row * col]();
	auto ArmLR = new int[row * col]();
	auto ArmLup = new int[row * col]();
	auto ArmLdown = new int[row * col]();

	//����洢�ұ۱۳�
	auto ArmRL = new int[row * col]();
	auto ArmRR = new int[row * col]();
	auto ArmRup = new int[row * col]();
	auto ArmRdown = new int[row * col]();

	// �۳��ռ�
	auto ArmVolumL = new int[row * col * dispRange]();
	auto ArmVolumR = new int[row * col * dispRange]();
	auto ArmVolumUp = new int[row * col * dispRange]();
	auto ArmVolumDown = new int[row * col * dispRange]();

	// �Ӳ�ռ�
	auto dispLeft = new float[row * col]();
	auto dispRight = new float[row * col]();
	
	// ������۳�,����ͼ����������صı۳�
	ArmLengthL(imageL, tao, ArmLL, maxLength, secLength);
	ArmLengthR(imageL, tao, ArmLR, maxLength, secLength);
	ArmLengthUp(imageL, tao, ArmLup, maxLength, secLength);
	ArmLengthDown(imageL, tao, ArmLdown, maxLength, secLength);
	cout << " finished left arm lenght calculate! " << endl;
	int width = col;
	int height = row;

	auto bytes_left = new uint8[width * height * 3];
	auto bytes_right = new uint8[width * height * 3];
	auto disparity = new float32[uint32(width * height)]();
	
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			bytes_left[i * 3 * width + 3 * j] = leftColor.at<cv::Vec3b>(i, j)[0];
			bytes_left[i * 3 * width + 3 * j + 1] = leftColor.at<cv::Vec3b>(i, j)[1];
			bytes_left[i * 3 * width + 3 * j + 2] = leftColor.at<cv::Vec3b>(i, j)[2];
			bytes_right[i * 3 * width + 3 * j] = rightColor.at<cv::Vec3b>(i, j)[0];
			bytes_right[i * 3 * width + 3 * j + 1] = rightColor.at<cv::Vec3b>(i, j)[1];
			bytes_right[i * 3 * width + 3 * j + 2] = rightColor.at<cv::Vec3b>(i, j)[2];
		}
	}

	auto leftPtr = new uchar[width * height]();
	auto rightPtr = new uchar[width * height]();

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			leftPtr[i * col + j] = imageL.at<uchar>(i, j);
			rightPtr[i * col + j] = imageR.at<uchar>(i, j);
		}
	}
	// �����ұ۳�������ͼ����������صı۳� 
	ArmLengthL(imageR, tao, ArmRL, maxLength, secLength);
	ArmLengthR(imageR, tao, ArmRR, maxLength, secLength);
	ArmLengthUp(imageR, tao, ArmRup, maxLength, secLength);
	ArmLengthDown(imageR, tao, ArmRdown, maxLength, secLength);
	cout << " finished right arm length calculate!" << endl;

	// ѡ�����ұ۵Ľ�����
	//chooseArmLengthLeft(ArmLL, ArmLR, ArmRL, ArmRR, dispRange, ArmVolumL, row, col);
	//chooseArmLengthRight(ArmLL, ArmLR, ArmRL, ArmRR, dispRange, ArmVolumR, row, col);
	//chooseArmLengthUp(ArmLup, ArmLdown, ArmRup, ArmRdown, ArmRL, ArmRR, dispRange, ArmVolumUp, row, col);
	//chooseArmLengthDown(ArmLup, ArmLdown, ArmRup, ArmRdown, ArmRL, ArmRR, dispRange, ArmVolumDown, row, col);

	//cv::Mat check = cv::Mat(row, col, CV_8UC1);
	//for (int i = 0; i < row; i++)
	//{
	//	for (int j = 0; j < col; j++)
	//	{
	//		const int disp_ = ArmLdown[i * col + j];
	//		check.at<uchar>(i, j) = static_cast<uchar>(disp_);
	//	}
	//}

	// ��չ�߽�
	copyMakeBorder(imageL, imageL, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
	copyMakeBorder(imageR, imageR, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);

	// ��չ��ɫ�߽�  ��ʵ���ֱ����Ļ����о��ǿ��Բ�������ģ���Ϊ�����ǿɱ�ģ����Ҳ���ȡ���������ֵ��Ҳ����ȡ�������߽��ֵ
	copyMakeBorder(leftColor, leftColor, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
	copyMakeBorder(rightColor, rightColor, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);

	// �����Ӳ�
	//ComputeDispLeft(dispVolumLeft, imageL, imageR, winSize, dispRange, row, col);   // ����ۿռ�
	ComputeAD(col, row, dispRange, leftPtr, rightPtr, dispVolumLeft);
	ComputeADRight(col, row, dispRange, leftPtr, rightPtr, dispVolumRight);


	// ���˵��㷨
	//CrossAggregator crossAggre;
	//crossAggre.Initialize(col, row, 0, dispRange);
	//crossAggre.SetData(bytes_left,bytes_right, dispVolum);
	//crossAggre.SetParams(option.cross_L1, option.cross_L2, option.cross_t1, option.cross_t2);
	//crossAggre.Aggregate(4);
	//auto cost = crossAggre.get_cost_ptr();

	// ���۾ۺ�
	costAggregationV5(dispVolumRight, costVolumRight, ArmRL, ArmRR, ArmRup, ArmRdown, dispRange, row, col, winSize);
	costAggregationV5(dispVolumLeft, costVolumLeft, ArmLL, ArmLR, ArmLup, ArmLdown, dispRange, row, col, winSize);  
	//costAggregationVertical(costVolumLeft, costVolumLeftSec, ArmLL, ArmLR, ArmLup, ArmLdown, dispRange, row, col, winSize);
	costAggregationV5(costVolumLeft, costVolumLeftSec, ArmLL, ArmLR, ArmLup, ArmLdown, dispRange, row, col, winSize);
	costAggregationV5(costVolumRight, costVolumRightSec, ArmLL, ArmLR, ArmLup, ArmLdown, dispRange, row, col, winSize);
	// WTA
	ComputeDispOringin(costVolumLeftSec, dispLeft, dispRange, row, col);  // ���Ӵ��۾ۺ�ʱ�����Ӳ�ͼ
	ComputeDispOringin(costVolumRightSec, dispRight, dispRange, row, col);  // ���Ӵ��۾ۺ�ʱ�����Ӳ�ͼ

	int gate = 5;
	vector<pair<int, int>> occlusion;
	vector<pair<int, int>> mismatches;

	// ����
	//LeftRightConsistency(col, row, gate, dispLeft, dispRight, occlusion, mismatches);
	//RemoveSpeckles(dispLeft, col, row, 1, 50, Invalid_Float);
	//MedianFilter(dispLeft, dispLeft, col, row, 3);

	// ��ʽת��
	cv::Mat disp_matLeft = cv::Mat(row, col, CV_8UC1);
	cv::Mat disp_matRight = cv::Mat(row, col, CV_8UC1);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			const int disp_ = dispLeft[i * col + j];
			if (disp_ == 0)
			{
				disp_matLeft.data[i * col + j] = 0;
			}
			else
			{
				//disp_mat.data[i * col + j] = 2 * static_cast<uchar>(disp_);
				disp_matLeft.at<uchar>(i,j) = static_cast<uchar>(disp_);
			}
		}
	}

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			const int disp_ = dispRight[i * col + j];
			if (disp_ == 0)
			{
				disp_matRight.data[i * col + j] = 0;
			}
			else
			{
				//disp_mat.data[i * col + j] = 2 * static_cast<uchar>(disp_);
				disp_matRight.at<uchar>(i, j) = static_cast<uchar>(disp_);
			}
		}
	}

	normalize(disp_matLeft, disp_matLeft, 0, 255, NORM_MINMAX);
	disp_matLeft.convertTo(disp_matLeft, CV_8UC1);
	normalize(disp_matRight, disp_matRight, 0, 255, NORM_MINMAX);
	disp_matRight.convertTo(disp_matRight, CV_8UC1);

	imshow("dispLeft", disp_matLeft);
	imshow("dispRight", disp_matRight);
	//imshow("imageL", imageL);
	imwrite("dispLeft.png", disp_matLeft);
	imwrite("dispRight.png", disp_matRight);
	waitKey();
	return 0;
}