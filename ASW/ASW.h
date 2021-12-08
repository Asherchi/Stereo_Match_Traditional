#pragma once
#include <opencv2/opencv.hpp>	
#include <iostream>
#include <vector>
#include <omp.h>
//#include "BiliteralFilter.h"
using namespace std;
using namespace cv;

constexpr auto Invalid = 0xffff;


////////////////////////////
//��ȡ��˹ģ�壨�ռ�ģ�壩
///////////////////////////
void getGausssianMask(cv::Mat& Mask, cv::Size wsize, double spaceSigma = 10) {
	Mask.create(wsize, CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;

	for (int i = 0; i < h; ++i) {
		y = pow(i - center_h, 2);
		double* Maskdate = Mask.ptr<double>(i);
		for (int j = 0; j < w; ++j) {
			x = pow(j - center_w, 2);
			double g = exp(-(x + y) / (2 * spaceSigma * spaceSigma));
			Maskdate[j] = g;
			sum += g;
		}
	}
}


////////////////////////////
//��ȡɫ��ģ�壨ֵ��ģ�壩
///////////////////////////
void getColorMask(std::vector<double>& colorMask, double colorSigma = 30) {

	for (int i = 0; i < 256; ++i) {
		double colordiff = exp(-(i * i) / (2 * colorSigma * colorSigma));
		colorMask.push_back(colordiff);
	}
}

void ComputeWeigtColor(const Mat& window , Mat& colorWeight,const int winSize, const Mat& space, const vector<double> color, int colorSigma = 30)
{
	// ����һ��С�����ڵ�LabȨ�أ�Ȼ��洢��colorWeight��
	int row = window.rows;
	int col = window.cols;
	float gamac = 5;
	int pointX = winSize;
	int pointY = winSize;
	//std::vector<double> colorMask;
	//getColorMask(colorMask, colorSigma);//ֵ��ģ��
	// ��ȡ�ο�����Ҳ�����������ص�ֵ
	auto central_L = window.at<Vec3b>(pointX, pointY)[0]; 
	auto central_a = window.at<Vec3b>(pointX, pointY)[1];
	auto central_b = window.at<Vec3b>(pointX, pointY)[2];

	// �������е�λ��ȥ�������������ص�ŷ����þ���
#pragma omp parallel for  
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto sur_L = window.at<Vec3b>(i, j)[0];
			auto sur_a = window.at<Vec3b>(i, j)[1];
			auto sur_b = window.at<Vec3b>(i, j)[2];
			float L = exp(-pow(central_L-sur_L,2)/(2*colorSigma*colorSigma));
			float A = exp(-pow(central_a - sur_a, 2) / (2 * colorSigma * colorSigma));
			float B = exp(-pow(central_b - sur_b, 2) / (2 * colorSigma * colorSigma));
			float mean = color[int((L + A + B) / 3)];
			colorWeight.at<float>(i, j) = mean;
		}
	}
}

void ComputeProximity(const Mat& window, Mat& colorWeight,const int winSize, const Mat& space, const vector<double> color, int spaceSigma = 30)
{
	// ������������֮���Զ����ϵ����ʵ����ŷ����þ���
	int row = window.rows;
	int col = window.cols;
	float gammap = 17.5;
	int pointX = winSize;
	int pointY = winSize;
	//cv::Mat spaceMask;

	//getGausssianMask(spaceMask, cv::Size(winSize*2+1,winSize*2+1), spaceSigma);//�ռ�ģ��
	// �����������أ���ʵ��������Ƕ���ģ����������˼���������Ϊ�����Сһ��ȷ��
	// ���е����ؾ��붼��ȷ���ģ�������ʵ�ܶ��࣬�����㷨�Ľ����Կ��Ǹĵ����
//#pragma omp parallel for  
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			//auto surPoint = window.at<int>(i, j);  // 
			//auto Gpq = sqrt(pow(pointX - i, 2)+ pow(pointY - j, 2));
			colorWeight.at<float>(i, j) = space.at<double>(i, j);
		}
	}
}


void CrossCheckDiaparity(const Mat& leftdisp, const Mat& rightdisp, Mat& lastdisp,const int MaxDisparity, const int winsize)
{
	int row = leftdisp.rows;
	int col = rightdisp.cols;
	int w = winsize;
	int rowrange = row;
	int colrange = col;
	int diffthreshold = 5;
	//occlusion.clear();
	//mismatch.clear();
#pragma omp parallel for  
	for (int i = 0; i < row; ++i)
	{
		const float* ptrleft = leftdisp.ptr<float>(i);  // ���������ĳһ�е��Ӳ�ֵ
		const float* ptrright = rightdisp.ptr<float>(i);
		uchar* ptrdisp = lastdisp.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			int leftvalue = *(ptrleft + j);  // ������������������ʲô����һ�����꣬�൱��ȡ��ĳһ�еĵ�i�е�����
			float rightvalue = *(ptrright + j - leftvalue);  // ΪʲôҪ��ȥ���Ӳ�ֵ�أ���Ϊ��ͼƥ����ͼ��ʱ���ҵ������ص�λ�øպò���Ӳ�ֵ������ͼ��������
			float diff = abs(leftvalue - rightvalue);  //�õ���Ӧ�Ӳ�ֵ֮�󣬾Ϳ�����һ���Լ�����
			if (diff > diffthreshold)
			{
				/*if (leftvalue < rightvalue)
					occlusion.emplace_back(i, j);
				else
					mismatch.emplace_back(i, j);*/
				*(ptrdisp + j) = 0;
			}
			else
			{
				*(ptrdisp + j) = uchar(leftvalue);
			}
		}
		//cout << "the process is" << static_cast<float>(i / row)*100 << "%" << endl;
	}

}


float ComputeCost(Mat& leftWin, Mat& rightWin, float trunc, const int winSize,const Mat& leftWinLab, const Mat& rightWinLab,
	const Mat& space, const vector<double> color)
{
	// �����ҵ�һ����������Ϊ�ǽض�SAD�����Լ���һ���ж�����
	float cost = 0;
	int row = leftWin.rows;
	int col = leftWin.cols;
	Mat weightColorLeft = Mat::zeros(row, col,CV_32FC1);
	Mat weightProximityLeft = Mat::zeros(row, col, CV_32FC1);
	Mat weightColorRight = Mat::zeros(row, col, CV_32FC1);
	Mat weightProximityRight = Mat::zeros(row, col, CV_32FC1);

	// ���㴰���ڵ�Ȩ�أ�������������
	ComputeWeigtColor(leftWinLab, weightColorLeft, winSize, space, color);
	ComputeWeigtColor(rightWinLab, weightColorRight, winSize,space,color);
	ComputeProximity(leftWin, weightProximityLeft, winSize, space, color);
	ComputeProximity(rightWin, weightProximityRight, winSize, space, color);
	Mat weight_ = weightColorLeft.mul(weightProximityLeft);
	Mat weight__ = weightColorRight.mul(weightProximityRight);
	Mat weightTotal = weight_.mul(weight__);
	Mat sub = cv::abs(leftWin-rightWin);
	sub.convertTo(sub, CV_32FC1);
	Mat E = weightTotal.mul(sub);
	cost = sum(E)[0]/(sum(weightTotal)[0]);
	if (cost > trunc)
		return trunc;
	return cost;
}

float ComputeCost(const Mat& left, const Mat& right)
{
	Mat subvalue;
	absdiff(left, right, subvalue);
	float value = sum(subvalue)[0];
	return value;
}

//Mat ComputeCost(const Mat& left, const Mat& right)
//{
//	Mat subvalue;
//	absdiff(left, right, subvalue);
//	//float value = sum(subvalue)[0];
//	return value;
//}

float WinTakeAll(vector<float> costVolume)
{
	// ����Ƚϼ򵥣�����һ���򵥵�Ӯ��ͨ��
	float minValue = costVolume[0];
	float disparity = 0;
#pragma omp parallel for  
	for (int i = 1; i < costVolume.size(); i++)
	{
		if (minValue > costVolume[i])
		{
			disparity = i;
			minValue = costVolume[i];
		}
	}
	return disparity;
}

float bilateralfiterWight(cv::Mat& src, cv::Mat& target, cv::Mat& error,  int wsize, const Mat& spaceMask, vector<double> colorMask)
{
	cv::Mat Mask0 = cv::Mat::zeros(src.rows, src.cols, CV_64F);
	cv::Mat Mask1 = cv::Mat::zeros(src.rows, src.cols, CV_64F);
	cv::Mat Mask2 = cv::Mat::zeros(src.rows, src.cols, CV_64F);
	int center_x = wsize + 1;  // wsize �ǰ뾶
	int center_y = wsize + 1;
	//dst.create(src.size(), src.type());

	double sum[3] = { 0 };   // Ϊɶ�����������������İ� �� ����ͨ��
	int graydiff[3] = { 0 };
	double space_color_sum[3] = { 0.0 };
	int centerPix = src.at<uchar>(center_x, center_y);
	for (int i = 0; i < src.rows; ++i) 
	{
		for (int j = 0; j < src.cols; ++j)
		{
				int pix = src.at<uchar>(i, j);
				graydiff[0] = abs(pix - centerPix);
				double colorWeight = colorMask[graydiff[0]];
				Mask0.at<double>(i, j) = colorWeight * spaceMask.at<double>(i, j);//�˲�ģ�� 
		}
	}

    //sum[0] = { 0 };   // Ϊɶ�����������������İ� �� ����ͨ��
	graydiff[0] = { 0 };
	//space_color_sum[3] = { 0.0 };
	centerPix = target.at<uchar>(center_x, center_y);
	for (int i = 0; i < target.rows; ++i)
	{
		for (int j = 0; j < target.cols; ++j)
		{
			int pix = target.at<uchar>(i, j);
			graydiff[0] = abs(pix - centerPix);
			double colorWeight = colorMask[graydiff[0]];
			Mask1.at<double>(i, j) = colorWeight * spaceMask.at<double>(i, j);//�˲�ģ�� 
		}
	}
	Mask2 = Mask0.mul(Mask1);
	auto sumWeight0 = cv::sum(Mask0)[0];
	auto sumWeight1 = cv::sum(Mask1)[0];
	//auto sumweight = sumWeight0 + sumWeight1;
	auto sumweight = cv::sum(Mask2)[0];
	error.convertTo(error, CV_64F);
	auto sumValue = Mask2.mul(error);
	auto finallValue = cv::sum(sumValue)[0] / sumweight;
	return finallValue;
}


void bilateralfiter(cv::Mat& src, cv::Mat& target, int winsize, const Mat& spaceMask, vector<double> colorMask)
{

	int C = (winsize-1)/2 ;
	int R = C;
	Mat paded;

	copyMakeBorder(src, paded, C, C, C, C, BORDER_REPLICATE);
	int col = src.cols - C;
	int row = src.rows- C;

	cv::Mat Mask0 = cv::Mat::zeros(winsize, winsize, CV_64F);
	cv::Mat Mask1 = cv::Mat::zeros(winsize, winsize, CV_64F);
	cv::Mat Mask2 = cv::Mat::zeros(winsize, winsize, CV_64F);

	for (int i = C; i < col; i++)
	{
		for (int j = C; j < row; j++)
		{
			double colorWeight[3] = { 0.0 };
			double sum[3] = { 0.0 };
			auto center = paded.at<Vec3b>(i, j);
			double allSum[3] = { 0.0 };
			for (int  a = -R; a <=R; a++)
			{
				for (int  b = -C; b <=C; b++)
				{
					auto pixel = paded.at<Vec3b>(a+i, b+j);
					int weight0 = abs(pixel[0] - center[0]);
					int weight1 = abs(pixel[1] - center[1]);
					int weight2 = abs(pixel[2] - center[2]);
					colorWeight[0] = colorMask[weight0] * spaceMask.at<double>(a + i, b + j);
					colorWeight[1] = colorMask[weight1] * spaceMask.at<double>(a + i, b + j);
					colorWeight[2] = colorMask[weight2] * spaceMask.at<double>(a + i, b + j);
					Mask0.at<uchar>(a + i, b + j) = colorWeight[0];
					Mask1.at<uchar>(a + i, b + j) = colorWeight[1];
					Mask2.at<uchar>(a + i, b + j) = colorWeight[1];
					sum[0] = sum[0]+ colorWeight[0];
					sum[1] = sum[1] + colorWeight[1];
					sum[2] = sum[2] + colorWeight[2];
				}
			}
			// ��һ��
			Mask0 = Mask0 / sum[0];
			Mask1 = Mask1 / sum[1];
			Mask2 = Mask2 / sum[2];

			//��ֵ
			for (int a = -R; a <= R; a++)
			{
				for (int b = -C; b <= C; b++)
				{
					auto value = paded.at<Vec3b>(i + a, b + j);
					allSum[0] =allSum[0]+  Mask0.at<uchar>(i + a, b + j) * value[0];
					allSum[1] = allSum[1] + Mask1.at<uchar>(i + a, b + j) * value[1];
					allSum[2] = allSum[2] + Mask2.at<uchar>(i + a, b + j) * value[2];
				}
			}
			for (int i = 0; i < 3; i++)
			{
				if (allSum[i] > 255)
					allSum[i] = 255;
			}
			cv::Vec3b bgr = { static_cast<uchar>(allSum[0]), static_cast<uchar>(allSum[1]), static_cast<uchar>(allSum[2]) };
			target.at<cv::Vec3b>(i - winsize, j - winsize) = bgr;
		}
	}
}

void AdaptiveSupportWeight(Mat &disp, const Mat& leftGray, const Mat& rightGray,const int winSize, int dispRange, const Mat& leftLab, 
	const Mat & rightLab, const Mat& space, const vector<double> color, int T)
{
	// ����������
	int wins = winSize + 1;
	int col = leftGray.cols-winSize-1;
	int row = leftGray.rows-winSize-1;
#pragma omp parallel for  
	for (int i = wins; i < row; i++)
	{
		float* ptr = disp.ptr<float>(i- wins);
		for (int j = wins; j < col; j++)
		{
			Mat leftWin = leftGray(Range(i - wins, i + wins + 1), Range(j - wins, j + wins + 1));
			Mat leftWinLab = leftLab(Range(i - wins, i + wins + 1), Range(j - wins, j + wins + 1));
			
			vector<float> costVolume(dispRange);
			for (int d = 0; d < dispRange; d++)
			{
				if (j - wins - d >= 0)
				{
					Mat rightWin = rightGray(Range(i - wins, i + wins + 1), Range(j - wins - d, j + wins - d + 1));
					Mat rightWinLab = rightLab(Range(i - wins, i + wins + 1), Range(j - wins - d, j + wins - d + 1));
					//costVolume[d] = ComputeCost(leftWin, rightWin, 40, wins, leftWinLab, rightWinLab, space, color);
					//Mat left, right;
					//bilateralfiterWight(leftWin, left, winSize, space, color);
					//bilateralfiterWight(rightWin, right, winSize, space, color);
					//costVolume[d] = ComputeCost(left, right) + ComputeCost(leftWin, rightWin);
					Mat error;
					absdiff(leftWin, rightWin, error);
					for (int i = 0; i < error.rows; i++)
					{
						for (int j = 0; j < error.cols; j++)
						{
							if (error.at<uchar>(i, j) > T)
								error.at<uchar>(i, j) = T;
						}
					}
					costVolume[d] = bilateralfiterWight(leftWin, rightWin, error, winSize, space, color);
				}
				else
				{
					costVolume[d] = costVolume[d-1];
				}
			}
			*(ptr + j- wins) = WinTakeAll(costVolume);
		}
		cout << " process left finished ...." << float(i) / float(row) * 100 << "%" << endl;
	}
}



void AdaptiveSupportWeightRight(Mat& disp, const Mat& leftGray, const Mat& rightGray, const int winSize, int dispRange, const Mat& leftLab,
	const Mat& rightLab, const Mat& space, const vector<double> color, int T)
{
	// ����������
	int wins = winSize + 1;
	int col = leftGray.cols - winSize - 1;
	int row = leftGray.rows - winSize - 1;
#pragma omp parallel for  
	for (int i = wins; i < row; i++)
	{
		float* ptr = disp.ptr<float>(i - wins);
		for (int j = wins; j < col; j++)
		{
			Mat leftWin = rightGray(Range(i - wins, i + wins + 1), Range(j - wins, j + wins + 1));
			Mat leftWinLab = rightLab(Range(i - wins, i + wins + 1), Range(j - wins, j + wins + 1));

			vector<float> costVolume(dispRange);
			for (int d = 0; d < dispRange; d++)
			{
				if (j + wins + d+1 < col)  // ����ط�ע��һ�� �����׳���
				{
					Mat rightWin = leftGray(Range(i - wins, i + wins + 1), Range(j - wins + d, j + wins + d + 1));
					Mat rightWinLab = leftLab(Range(i - wins, i + wins + 1), Range(j - wins + d, j + wins + d + 1));
					Mat error;
					absdiff(leftWin, rightWin, error);
					for (int i = 0; i < error.rows; i++)
					{
						for (int j = 0; j < error.cols; j++)
						{
							if (error.at<uchar>(i, j) > T)
								error.at<uchar>(i, j) = T;
						}
					}
					costVolume[d] = bilateralfiterWight(leftWin, rightWin, error, winSize, space, color);
					//costVolume[d] = ComputeCost(leftWin, rightWin, 0.4, wins, leftWinLab, rightWinLab, space, color);
					//Mat left, right;
					//bilateralfiterWight(leftWin, left, winSize, space, color);
					//bilateralfiterWight(rightWin, right, winSize, space, color);
					//costVolume[d] = ComputeCost(left, right);
				}
				else
				{
					costVolume[d] = costVolume[d - 1];
				}
			}
			*(ptr + j - wins) = WinTakeAll(costVolume);
		}
		cout << " process right finished ...." << float(i) / float(row) * 100 << "%" << endl;
	}
}


void FillImageNew(Mat& disp)
{
	int col = disp.cols;
	int row = disp.rows;
	vector<pair<int, int>> Nonvalue;
	vector<int> dispValue;

	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			auto value = disp.at<uchar>(i, j);
			if (value)
			{
				continue;
			}
			else
			{
				Nonvalue.emplace_back(i, j);
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixel = Nonvalue[i];
		int pixel_row = pixel.first;
		int pixel_col = pixel.second;
		int offset = 0;

		if (pixel_col >= 0)
		{
			while (pixel_col >= 0)
			{
				int x = pixel_row;
				pixel_col = pixel_col - offset;
				if (pixel_col < 0)
				{
					pixel_col = pixel.second;  // �����趨��ʼֵ
					offset = 0;
					//dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(x, pixel_col);
				if (value)
				{
					dispValue.push_back(value);
					pixel_col = 0xffff;  // Ϊ��ʹ�����if��ִ��
					break;
				}
				++offset;
			}
			while (pixel_col < col)
			{
				int x = pixel_row;
				pixel_col = pixel_col + offset;
				if (pixel_col > col)
				{
					dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(x, pixel_col);
				if (value)
				{
					dispValue.push_back(value);
					break;
				}
				++offset;
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixelN = Nonvalue[i];
		int pixel_row = pixelN.first;
		int pixel_col = pixelN.second;
		disp.at<uchar>(pixel_row, pixel_col) = dispValue[i];
	}
}


void FillImageLast(Mat& disp)
{
	int col = disp.cols;
	int row = disp.rows;
	vector<pair<int, int>> Nonvalue;
	vector<int> dispValue;

	for (auto i = 0; i < row; i++)
	{
		for (auto j = 0; j < col; j++)
		{
			auto value = disp.at<uchar>(i, j);
			if (value)
			{
				continue;
			}
			else
			{
				Nonvalue.emplace_back(i, j);
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixel = Nonvalue[i];
		int pixel_row = pixel.first;
		int pixel_col = pixel.second;
		int offset = 0;

		if (pixel_row >= 0)
		{
			while (pixel_row >= 0)
			{
				int y = pixel_col;
				pixel_row = pixel_row - offset;
				if (pixel_row < 0)
				{
					pixel_row = pixel.first;  // �����趨��ʼֵ
					offset = 0;
					//dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(pixel_row, y);
				if (value)
				{
					dispValue.push_back(value);
					pixel_row = 0xffff;  // Ϊ��ʹ�����if��ִ��
					break;
				}
				++offset;
			}
			while (pixel_row < row)
			{
				int y = pixel_col;
				pixel_row = pixel_row + offset;
				if (pixel_row > row)
				{
					dispValue.push_back(0);
					break;
				}
				auto value = disp.at<uchar>(pixel_row, y);
				if (value)
				{
					dispValue.push_back(value);
					break;
				}
				++offset;
			}
		}
	}
	for (auto i = 0; i < Nonvalue.size(); i++)
	{
		auto pixelN = Nonvalue[i];
		int pixel_row = pixelN.first;
		int pixel_col = pixelN.second;
		disp.at<uchar>(pixel_row, pixel_col) = dispValue[i];
	}
}

