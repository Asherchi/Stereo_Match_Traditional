#pragma once
#include<iostream>	
#include<opencv2/opencv.hpp>
#include <iomanip>
#include <numeric>

using namespace std;
using namespace cv;

float sadvalue(const Mat& src1, const Mat& src2)
{
	Mat  matdiff = cv::abs(src1 - src2);
	float  saddiff = cv::sum(matdiff)[0];
	return saddiff;
}

float sadvalueMean(const Mat& src1, const Mat& src2)
{
	Mat  matdiff = cv::abs(src1 - src2);
	float value = cv::mean(matdiff)[0];
	return value;
}

uchar minValue(uchar a, uchar b, uchar c, int T)
{
	if (a > b)
	{
		if (b > c)
			return c;
		else
		{
			return b;
		}
	}
	else
	{
		if (a > c)
			return c;
		else
		{
			return a;
		}
	}
}

float sadvalueMeanV4(const Mat& src1, const Mat& src2, int T = 60)
{
	Mat  matdiff;
	absdiff(src1, src2, matdiff);
	uchar sumvalue = 0, value = 0;
	int number = 0;
	for (auto i = 0; i < matdiff.rows; i++)
	{
		for (int j = 0; j < matdiff.cols; j++)
		{
			sumvalue += minValue(matdiff.at<Vec3b>(i,j)[0], matdiff.at<Vec3b>(i, j)[1], matdiff.at<Vec3b>(i, j)[2], T);
			number++;
		}
	}
	value = sumvalue / number;
	//float value = cv::mean(matdiff)[0];
	return value;
}

void chooseArmLengthLeft(int* ArmLL, int* ArmLR, int* ArmRL, int* ArmRR, int dispRange, int* Armvolume, int row, int col)
{
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			for (int d = 0; d < dispRange; d++)
			{
				int saveValue = 0;
				auto LL = ArmLL[i * col + j];
				auto RL = ArmRL[i * col + j];
				auto RR = ArmRR[i * col + j];
				if ((j -d < j - RL) || (j + d > j + RR))  //　首先判断中心点满不满足条件，不满足直接pass 
				{
					saveValue = 0;  // 稳妥起见
					auto& ArmV = Armvolume[i * col * dispRange + j * dispRange + d];
					ArmV = saveValue;
					continue; // 直接跳到下一个运算；
				}
				else
				{
					for (int a = 1; a <= LL; a++)
					{
						if (((j - a - d) >= (j - RL)) && ((j - a - d) <= (j + RR)))  // j-RL 是右图左边界， j+RR 是右图右边界
							saveValue++;
						else
						{
							break;
						}
					}
				}
				auto& ArmV = Armvolume[i * col * dispRange + j * dispRange + d];
				ArmV = saveValue;
				saveValue = 0; // 稳妥起见  
			}
		}
	}
}

void chooseArmLengthRight(int* ArmLL, int* ArmLR, int* ArmRL, int* ArmRR, int dispRange, int* Armvolume, int row, int col)
{
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			for (int d = 0; d < dispRange; d++)
			{
				int saveValue = 0;
				auto LR = ArmLR[i * col + j];
				auto RL = ArmRL[i * col + j];
				auto RR = ArmRR[i * col + j];
				//if (j + d > col)  // 我感觉这一步可以不做的，因为下面的也会对j+d进行一个判断
				//{
				//	saveValue = 0;  // 稳妥起见
				//	auto& ArmV = Armvolume[i * col * dispRange + j * dispRange + d];
				//	ArmV = saveValue;
				//	continue;
				//}
				if ((j - d < j - RL) || (j - d > j + RR))  //　首先判断中心点满不满足条件，不满足直接pass 
				{
					saveValue = 0;  // 稳妥起见
					auto& ArmV = Armvolume[i * col * dispRange + j * dispRange + d];
					ArmV = saveValue;
					continue; // 直接跳到下一个运算；
				}
				else
				{
					for (int a = 1; a <= LR; a++)
					{
						if ((j + a - d >= j - RL) && (j + a - d < j + RR))
							saveValue++;
						else
						{
							break;
						}
					}
				}
				auto& ArmV = Armvolume[i * col * dispRange + j * dispRange + d];
				ArmV = saveValue;
				saveValue = 0; // 稳妥起见  
			}
		}
	}
}


void chooseArmLengthUp(int* ArmLUp, int* ArmLDown,  int* ArmRUp, int * ArmRDown, int* ArmRL, int* ArmRR, int dispRange, int* Armvolume, int row, int col)
{
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			for (int d = 0; d < dispRange; d++)
			{
				int saveValue = 0;
				auto LUp = ArmLUp[i * col + j];   // 左臂上	
				auto LDown = ArmLDown[i * col + j];  // 左臂下
				auto RUp = ArmRUp[i * col + j]; // 右臂上
				auto RDown = ArmRDown[i * col + j]; // 右臂下
				for (int up = 1; up <= LUp; up++)
				{
					int pointrow = i - up;
					int ptrL = ArmRL[pointrow * col + j];  // 右图偏移点的左右臂长
					int ptrR = ArmRR[pointrow * col + j];
					if (pointrow >= (i - RUp))  // 使得左图点在右图右同名点  如果右上的值比左上的值大的话才能进行判断，否则不进行判断
					{
						if (j-d < 0)
						{
							break;
						}
						if ((( j - d ) < (j + ptrR)) && (( j - d ) > (j - ptrL)))
						{
							saveValue++;
						}
					}
					else
					{
						saveValue = 0;
						break; // 这个不能用break
					}
				}
				auto& ArmV = Armvolume[i * col * dispRange + j * dispRange + d];
				ArmV = saveValue;
				saveValue = 0; // 稳妥起见  
			}
		}
	}
}


void chooseArmLengthDown(int* ArmLUp, int* ArmLDown, int* ArmRUp, int* ArmRDown, int* ArmRL, int* ArmRR, int dispRange, int* Armvolume, int row, int col)
{
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			for (int d = 0; d < dispRange; d++)
			{
				int saveValue = 0;
				auto LUp = ArmLUp[i * col + j];   // 左臂上	
				auto LDown = ArmLDown[i * col + j];  // 左臂下
				auto RUp = ArmRUp[i * col + j]; // 右臂上
				auto RDown = ArmRDown[i * col + j]; // 右臂下
				for (int down = 1; down <= LDown; down++)
				{
					int pointrow = i + down;
					int ptrL = ArmRL[pointrow * col + j];  // 右图偏移点的左右臂长
					int ptrR = ArmRR[pointrow * col + j];
					if (pointrow <= i + RDown)  // 使得左图点在右图右同名点
					{
						if (j - d < 0)
						{
							saveValue = 0;
							break;
						}
						if ((j - d <=j + ptrR) && (j - d >= j - ptrL))
						{
							saveValue++;
						}
					}
					else
					{
						break; // 这个不能用break
					}
				}
				auto& ArmV = Armvolume[i * col * dispRange + j * dispRange + d];
				ArmV = saveValue;
				saveValue = 0; // 稳妥起见  
			}
		}
	}
}

int maxVec(Vec3f input)
{
	int maxValue = input[0];
	if (maxValue < input[1])
		maxValue = input[1];
	if (maxValue < input[2])
		maxValue = input[2];
	return maxValue;
}


float OptimalDisparity(std::vector<float>& sad, const int disp_range)
{
	float minDisp_value = 0xffff;
	float best_disp = 0;
	float secMin = sad[0];
#pragma omp parallel for  
	for (auto i = 1; i < sad.size(); ++i)
	{
		if (minDisp_value > sad[i])
		{
			minDisp_value = sad[i];
			best_disp = i;
		}
	}

	for (auto i = 0; i < sad.size(); ++i)
	{
		if (minDisp_value == sad[i])
			continue;
		else
		{
			const auto cost = sad[i];
			secMin = min(secMin, cost);
		}
	}

	if (secMin - minDisp_value <= 0.01)
	{
		return 0;
	}

	if (best_disp == 0 || best_disp == disp_range - 1)
	{
		return 0;
	}

	const int index1 = best_disp - 1;
	const int index2 = best_disp + 1;
	const float cost1 = sad[index1];
	const float cost2 = sad[index2];
	const float divided = max(static_cast<float>(1), cost1 + cost2 - 2 * minDisp_value);
	const float final_disp = best_disp + (cost1 - cost2) / (2 * divided);
	//cout << " best disp is:" << best_disp << endl;

	return best_disp;
}

void ComputeDisp(int* dispVolume, float* costVolume, float* disp, int dispRange, int _row_, int _col_)
{
	int col = _col_;
	int row = _row_;
#pragma omp parallel for  
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			auto mindisp = dispVolume[i * col * dispRange + j * dispRange+ 0];
			auto minCost = costVolume[i * col * dispRange + j * dispRange + 0];
			float minValue = mindisp + minCost;
			auto bestDisp = 0;
			for (int d = 0; d < dispRange; ++d)
			{
				auto disp = dispVolume[i * col * dispRange + j * dispRange + d];
				auto cost = costVolume[i * col * dispRange + j * dispRange + d];
				auto costValue = disp + cost;
				if (costValue < minValue)
				{
					minValue = costValue;
					bestDisp = d;
				}
			}
			auto& dispC = disp[i * col + j];
			dispC = bestDisp;
			
		}
	}
}

void ComputeAD(int __col__, int __row__, int __dispRange__, uchar* __leftImage__, uchar* __rightImage__, float* ADcostVolum)
{
	int row = __row__;
	int col = __col__;
	int dispRange = __dispRange__;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto leftValue = __leftImage__[i * col + j];
			for (int d = 0; d < dispRange; d++)
			{
				auto& cost = ADcostVolum[i * col * dispRange + j * dispRange + d];
				if (j - d < 0)
				{
					const float value = ADcostVolum[i * col * dispRange + j * dispRange + d - 1];
					cost = value;
				}
				else
				{
					auto rightValue = __rightImage__[i * col + j - d];
					cost = abs(leftValue - rightValue);
				}
			}
		}
	}
}

void ComputeADRight(int __col__, int __row__, int __dispRange__, uchar* __leftImage__, uchar* __rightImage__, float* ADcostVolum)
{
	int row = __row__;
	int col = __col__;
	int dispRange = __dispRange__;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto rightValue = __rightImage__[i * col + j];
			for (int d = 0; d < dispRange; d++)
			{
				auto& cost = ADcostVolum[i * col * dispRange + j * dispRange + d];
				if (j + d >= col)
				{
					const float value = ADcostVolum[i * col * dispRange + j * dispRange + d - 1];
					cost = value;
				}
				else
				{
					auto leftValue = __leftImage__[i * col + j + d];
					cost = abs(leftValue - rightValue);
				}
			}
		}
	}
}

void ComputeDispOringin(float* costVolume, float* disp, int dispRange, int _row_, int _col_)
{
	int col = _col_;
	int row = _row_;
//#pragma omp parallel for  
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			auto minCost = costVolume[i * col * dispRange + j * dispRange + 0];
			auto bestDisp = 0;
			for (int d = 0; d < dispRange; ++d)
			{
				auto cost = costVolume[i * col * dispRange + j * dispRange + d];
				if (cost < minCost)
				{
					minCost = cost;
					bestDisp = d;
				}
			}
			auto& dispC = disp[i * _col_ + j];
			dispC = bestDisp;
		}
	}
}

void ComputeDispRight(float* dispVolume, const Mat& leftimg, const Mat& rightimg, int winsize, int dispRange, int _row_, int _col_)
{
	int row = leftimg.rows;
	int col = leftimg.cols;
	if (leftimg.channels() == 3 && rightimg.channels() == 4)
	{
		cvtColor(leftimg, leftimg, CV_BGR2GRAY);
		cvtColor(rightimg, rightimg, CV_BGR2GRAY);
	}

	int win_r = winsize + 1;   // 这里为什么加1减一，画个图就明白了
	int rowrange = row - winsize - 1;
	int colrange = col - winsize - 1;
#pragma omp parallel for  
	for (int i = win_r; i < rowrange; ++i)
	{
		for (int j = win_r; j < colrange; ++j)  // 这里控制了从最大视差位置开始，会导致图像变小
		{
			Mat rightwin = rightimg(Range(i - win_r, i + win_r + 1), Range(j - win_r, j + win_r + 1));
			std::vector<float> sad(dispRange);
			for (int d = 0; d < dispRange; ++d)  // 获取左右的视差的方法完全不一样
			{
				int row_ = i - win_r;
				int col_ = j - win_r;
				auto& cost = dispVolume[row_ * _col_ * dispRange + col_ * dispRange + d];
				if (j + d + win_r + 1> colrange)
				{
					const auto last = dispVolume[row_ * _col_ * dispRange + col_ * dispRange + d -1]; // 注意这里的行列不能搞错
					cost = last;
					continue;
				}
				Mat leftwin = leftimg(Range(i - win_r, i + win_r + 1), Range(j - win_r + d, j + d + win_r + 1));
				cost = sadvalueMean(leftwin, rightwin);
			}
		}
		double rate = double(i) / (colrange);  // showing the process
		cout << "Depth Right finished" << setprecision(2) << rate * 100 << "%" << endl;
	}
}



void ComputeDispLeft(float* dispVolume, const Mat& leftimg, const Mat& rightimg, int winsize, int dispRange, int _row_, int _col_)
{
	int row = leftimg.rows;
	int col = leftimg.cols;
	if (leftimg.channels() == 3 && rightimg.channels() == 4)
	{
		cvtColor(leftimg, leftimg, CV_BGR2GRAY);
		cvtColor(rightimg, rightimg, CV_BGR2GRAY);
	}

	int win_r = winsize + 1;   // 这里为什么加1减一，画个图就明白了
	int rowrange = row - winsize - 1;
	int colrange = col - winsize - 1;
#pragma omp parallel for  
	for (int i = win_r; i < rowrange; ++i)
	{
		for (int j = win_r; j < colrange; ++j)  // 这里控制了从最大视差位置开始，会导致图像变小
		{
			Mat leftwin = leftimg(Range(i - win_r, i + win_r + 1), Range(j - win_r, j + win_r + 1));
			std::vector<float> sad(dispRange);
			for (int d = 0; d < dispRange; ++d)  // 获取左右的视差的方法完全不一样
			{
				int row_ = i - win_r;
				int col_ = j - win_r;
				auto& cost = dispVolume[row_ * _col_ * dispRange + col_ * dispRange + d];
				if (j - win_r - d < 0)
				{
					const auto last = dispVolume[row_ * _col_ * dispRange + col_ * dispRange + d - 1]; // 注意这里的行列不能搞错
					cost = last;
					continue;
				}
				Mat rightwin = rightimg(Range(i - win_r, i + win_r + 1), Range(j - win_r - d, j - d + win_r + 1));
				cost = sadvalueMean(leftwin, rightwin);
			}
		}
		double rate = double(i) / (colrange);  // showing the process
		cout << "Depth left finished" << setprecision(2) << rate * 100 << "%" << endl;
	}
}




void ComputeDispV4(float* dispVolume, const Mat& leftimg, const Mat& rightimg, int winsize, int dispRange, int _row_, int _col_)
{
	int row = leftimg.rows;
	int col = leftimg.cols;
	//if (leftimg.channels() == 3 && rightimg.channels() == 4)
	//{
	//	cvtColor(leftimg, leftimg, CV_BGR2GRAY);
	//	cvtColor(rightimg, rightimg, CV_BGR2GRAY);
	//}

	int win_r = winsize + 1;   // 这里为什么加1减一，画个图就明白了
	int rowrange = row - winsize - 1;
	int colrange = col - winsize - 1;
#pragma omp parallel for  
	for (int i = win_r; i < rowrange; ++i)
	{
		for (int j = win_r; j < colrange; ++j)  // 这里控制了从最大视差位置开始，会导致图像变小
		{
			Mat leftwin = leftimg(Range(i - win_r, i + win_r + 1), Range(j - win_r, j + win_r + 1));
			std::vector<float> sad(dispRange);
			for (int d = 0; d < dispRange; ++d)  // 获取左右的视差的方法完全不一样
			{
				int row_ = i - win_r;
				int col_ = j - win_r;
				auto& cost = dispVolume[row_ * _col_ * dispRange + col_ * dispRange + d];
				if (j - win_r - d < 0)
				{
					const auto last = dispVolume[row_ * _col_ * dispRange + col_ * dispRange + d - 1]; // 注意这里的行列不能搞错
					cost = last;
					continue;
				}
				Mat rightwin = rightimg(Range(i - win_r, i + win_r + 1), Range(j - win_r - d, j - d + win_r + 1));
				cost = sadvalueMeanV4(leftwin, rightwin);
			}
		}
		double rate = double(i) / (colrange);  // showing the process
		cout << "Depth left finished" << setprecision(2) << rate * 100 << "%" << endl;
	}
}



void ArmLengthR(const Mat& leftImage, uchar tao, int *RArm, int maxLength, int secLength)  // right
{
	const int row = leftImage.rows;
	const int col = leftImage.cols;
    int savedNumber = 0;
	if (leftImage.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = leftImage.at<Vec3b>(i, j)[0];
				auto y = leftImage.at<Vec3b>(i, j)[1];
				auto z = leftImage.at<Vec3b>(i, j)[2];
				while (j + offset < col)
				{
					savedNumber = offset;
					offset++;
					if (offset > secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (j + offset < col)
					{
						auto x_ = leftImage.at<Vec3b>(i, j + offset)[0];
						auto y_ = leftImage.at<Vec3b>(i, j + offset)[1];
						auto z_ = leftImage.at<Vec3b>(i, j + offset)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > tao)
						{
							if ((j + 1 < col-1) && (savedNumber <1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					if(j+offset>=col)
					{
						break;
					}
				}
				RArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				auto x = leftImage.at<uchar>(i, j);
				while (j + offset < col)
				{
					savedNumber = offset;
					offset++;
					if (offset > secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (j + offset < col)
					{
						auto x_ = leftImage.at<uchar>(i, j + offset);
						auto maxValue = std::abs(x - x_);
						if (maxValue > tao)
						{
							if ((j + 1 < col-1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					if(j+offset>=col)
					{
						break;
					}
				}
				RArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}



void ArmLengthL(const Mat& leftImage, uchar tao, int* LArm, int maxLength, int secLength)  // right
{
	const int row = leftImage.rows;
	const int col = leftImage.cols;
	int savedNumber = 0;
//#pragma omp parallel for  
	if (leftImage.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = leftImage.at<Vec3b>(i, j)[0];
				auto y = leftImage.at<Vec3b>(i, j)[1];
				auto z = leftImage.at<Vec3b>(i, j)[2];
				while (j - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset> secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (j - offset >= 0)
					{
						//savedNumber = offset;
						auto x_ = leftImage.at<Vec3b>(i, j - offset)[0];
						auto y_ = leftImage.at<Vec3b>(i, j - offset)[1];
						auto z_ = leftImage.at<Vec3b>(i, j - offset)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > tao)
						{
							if ((j - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					if(j-offset<0)
					{
						//savedNumber = 1;
						break;
					}
				}
				LArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = leftImage.at<uchar>(i, j);
				while (j - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset > secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (j - offset >= 0)
					{
						auto x_ = leftImage.at<uchar>(i, j - offset);
						auto maxValue = std::abs(x - x_);
						if (maxValue > tao)
						{
							if ((j - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					if(j-offset<0)
					{
						break;
					}
				}
				LArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}


void ArmLengthUp(const Mat& leftImage, uchar tao, int* UpArm, int maxLength, int secLength)  // right
{
	const int row = leftImage.rows;
	const int col = leftImage.cols;
    int savedNumber = 0;
//#pragma omp parallel for  
	if (leftImage.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = leftImage.at<Vec3b>(i, j)[0];
				auto y = leftImage.at<Vec3b>(i, j)[1];
				auto z = leftImage.at<Vec3b>(i, j)[2];
				while (i - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset > secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (i - offset >= 0)
					{
						//savedNumber = offset;
						auto x_ = leftImage.at<Vec3b>(i - offset, j)[0];
						auto y_ = leftImage.at<Vec3b>(i - offset, j)[1];
						auto z_ = leftImage.at<Vec3b>(i - offset, j)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > tao)
						{
							if ((i - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					if(i-offset<0)
					{
						break;
					}
				}
				UpArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				auto x = leftImage.at<uchar>(i, j);
				while (i - offset >= 0)
				{
					savedNumber = offset;
					offset++;
					if (offset > secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (i - offset >= 0)
					{
						auto x_ = leftImage.at<uchar>(i - offset, j);
						auto maxValue = std::abs(x - x_);
						if (maxValue > tao)
						{
							if ((i - 1 >= 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					if(i-offset<0)
					{
						break;
					}
				}
				UpArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}


void ArmLengthDown(const Mat& leftImage, uchar tao, int* DownArm, int maxLength, int secLength)  // right
{
	const int row = leftImage.rows;
	const int col = leftImage.cols;
	int savedNumber = 0;
//#pragma omp parallel for  
	if (leftImage.channels() == 3)
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				//vector<uchar> diffValue(3);
				auto x = leftImage.at<Vec3b>(i, j)[0];
				auto y = leftImage.at<Vec3b>(i, j)[1];
				auto z = leftImage.at<Vec3b>(i, j)[2];
				while (i + offset < row)
				{
					savedNumber = offset;
					offset++;
					if (offset > secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (i + offset < row)
					{
						//savedNumber = offset;
						auto x_ = leftImage.at<Vec3b>(i + offset, j)[0];
						auto y_ = leftImage.at<Vec3b>(i + offset, j)[1];
						auto z_ = leftImage.at<Vec3b>(i + offset, j)[2];
						auto xValue = std::abs(x - x_);
						auto yValue = std::abs(y - y_);
						auto zValue = std::abs(z - z_);
						auto maxValue = max(max(xValue, yValue), zValue);
						if (maxValue > tao)
						{
							if ((i + 1 < row - 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					if(i+offset>=row)
					{
						break;
					}
				}
				DownArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				int offset = 0;
				auto x = leftImage.at<uchar>(i, j);
				while (i + offset < row)
				{
					savedNumber = offset;
					offset++;
					if (offset > secLength)
					{
						tao = 6;
						if (offset > maxLength)
							break;
					}
					if (i + offset < row)
					{
						auto x_ = leftImage.at<uchar>(i + offset, j);
						auto maxValue = std::abs(x - x_);
						if (maxValue > tao)
						{
							if ((i + 1 < row - 1) && (savedNumber < 1))
							{
								savedNumber = 1;
								break;
							}
							else
							{
								break;
							}
						}
					}
					else
					{
						break;
					}
				}
				DownArm[i * col + j] = savedNumber;
				savedNumber = 0;
			}
		}
	}
}


float ComputeLocalValue(int i, int j, int Up, int Down, int winsize, int* LArm, int* RArm, int _col_, const Mat&Image,  int dispRange, int d=0)
{
	//vector<float> HorizanSum(Up+Down+2);  // 防止溢出所以加了2 
	//auto& cost = CostVolume[ptri * _col_ * dispRange + ptrj * dispRange + d];
	auto begin = i - Up;
	//int ptr = 0;
	int ptri = i - winsize;
	int ptrj = j - winsize;
	int count = 0;
	float value = 0;
#pragma omp parallel for
	for (int _row = - Up; _row <=  Down; _row++)
	{
		int ptr_i = i - winsize + _row; 
		int L = LArm[ptr_i * _col_ * dispRange + ptrj*dispRange + d];  // 取出某一点的左臂长
		int R = RArm[ptr_i * _col_ * dispRange + ptrj*dispRange + d];  // 取出某一点的右臂长
		//cout << "when d is :  " << d << " the down left is: " << L << endl;
		Mat image;
		if (d>0)  // 右图计算区域面积
		{
			if (j - L - d < 0)
			{
				if (j+R-d<=0)
				{
					image = Image(Range(i + _row, i + _row + 1), Range(0, 1));  // 如果左右都是小于等于0的,则选择取单个像素
					count = count + 1; //  这里可能会有一点小误差，但是我感觉影响不大
					value = value + sum(image)[0];
					//auto sumValue = sum(image)[0]; // 判断大小	
					//if (sumValue > 60)
					//	sumValue = 60;
					//value = value + sumValue;
				}
				else
				{
					image = Image(Range(i + _row, i + _row + 1), Range(0, j + R - d));
					count = count + R+1; //  这里可能会有一点小误差，但是我感觉影响不大
					value = value + sum(image)[0];
					//auto sumValue = sum(image)[0]; // 判断大小	
					//if (sumValue > 60)
					//	sumValue = 60;
					//value = value + sumValue;
				}
			}
			else
			{
				image = Image(Range(i + _row, i + _row + 1), Range(j - L - d, j + R - d)); // 取出某一行的数据
				count = count + L + R+1; //  这里可能会有一点小误差，但是我感觉影响不大
				value = value + sum(image)[0];
				//auto sumValue = sum(image)[0]; // 判断大小	
				//if (sumValue > 60)
				//	sumValue = 60;
				//value = value + sumValue;
			}
		}
		else // 左图计算区域面积
		{
			if ((L==0)&&(R==0))
			{
				image = Image(Range(i + _row, i + _row + 1), Range(j - L - d, j + R - d+1)); // 取出某一行的数据  d = 0
			}
			else
			{
				image = Image(Range(i + _row, i + _row + 1), Range(j - L - d, j + R - d)); // 取出某一行的数据  d = 0
			}
			count = count + L + R+1; //  这里可能会有一点小误差，但是我感觉影响不大
			value = value + sum(image)[0];
			//auto sumValue = sum(image)[0]; // 判断大小	
			//if (sumValue > 60)
			//	sumValue = 60;
			//value = value + sumValue;
		}
	}
	value = value / count;
	//auto value = accumulate(HorizanSum.begin(), HorizanSum.end(), 0) / (HorizanSum.size()-1);  // 前面size加了2，现在减1减小误差
	//cost = value;
	return value;
}


void costAggregation(const Mat& leftImage, const Mat& rightImage, float* CostVolume, int* LArmL, int* RArmL, int* UpArmL, 
	int* DownArmL, int dispRange, int _row_, int _col_, int winSize, int *RArmR, int *LArmR, int *UpArmR, int *DownArmR)
{
	int winsize = winSize + 1;
	int col = leftImage.cols-winSize-1;
	int row = rightImage.rows-winSize-1;
	if (leftImage.channels() == 3 && leftImage.channels() == 4)
	{
		cvtColor(leftImage, leftImage, CV_BGR2GRAY);
		cvtColor(leftImage, leftImage, CV_BGR2GRAY);
	}
#pragma omp parallel for  
	for (int i = winsize; i < row; ++i)
	{
		int ptri = i - winsize;
		for (int j = winsize; j < col; ++j)
		{
			int ptrj = j - winsize;
			int UpL = UpArmL[ptri * _col_ + ptrj];
			int DownL = DownArmL[ptri * _col_ + ptrj];
			int UpR = UpArmR[ptri * _col_ + ptrj];
			int DownR = DownArmR[ptri * _col_ + ptrj];
			auto leftValue = ComputeLocalValue(i, j, UpL, DownL, winsize, LArmL, RArmL, _col_, leftImage, dispRange);
			//cout << "leftValue: " << leftValue << endl;
			for (int d = 0; d < dispRange; d++)
			{	
				auto rightValue = ComputeLocalValue(i, j, UpR, DownR, winsize, LArmR, RArmR, _col_, rightImage, d);
				auto value = abs(leftValue - rightValue);
				auto& cost = CostVolume[ptri * _col_ * dispRange + ptrj * dispRange + d];
				//if (value > 60)  // 这里设的是截断SAD 
				//	value = 60;
				cost = value;
			}
		}
		double rate = double(i) / (row);  // showing the process
		cout << "cost aggregation finishing " << setprecision(2) << rate * 100 << "%" << endl;
	}
}

void costAggregationNew(const Mat& leftImage, const Mat& rightImage, float* CostVolume, int* ArmvolumeL, int* ArmvolumeR, int* ArmvolumeUp,
	int* ArmvolumeDown, int dispRange, int _row_, int _col_, int winSize)
{
	int winsize = winSize + 1;
	int col = leftImage.cols - winSize - 1;
	int row = rightImage.rows - winSize - 1;
	if (leftImage.channels() == 3 && leftImage.channels() == 4)
	{
		cvtColor(leftImage, leftImage, CV_BGR2GRAY);
		cvtColor(leftImage, leftImage, CV_BGR2GRAY);
	}
#pragma omp parallel for  
	for (int i = winsize; i < row; ++i)
	{
		int ptri = i - winsize;
		for (int j = winsize; j < col; ++j)
		{
			//cout << "leftValue: " << leftValue << endl;
			for (int d = 0; d < dispRange; d++)
			{
				int ptrj = j - winsize;
				int Up = ArmvolumeUp[ptri * _col_ * dispRange + ptrj*dispRange+d];
				int Down = ArmvolumeDown[ptri * _col_ * dispRange + ptrj * dispRange + d];
				/*cout << "when d is :  " << d << " the down length is: " << Down << endl;*/
				//int UpR = UpArmR[ptri * _col_ + ptrj];
				//int DownR = DownArmR[ptri * _col_ + ptrj];
				auto leftValue = ComputeLocalValue(i, j, Up, Down, winsize, ArmvolumeL, ArmvolumeR, _col_, leftImage, dispRange);
				//
				auto rightValue = ComputeLocalValue(i, j, Up, Down, winsize, ArmvolumeL, ArmvolumeR, _col_, rightImage,  dispRange, d);
				auto value = abs(leftValue - rightValue);
				auto& cost = CostVolume[ptri * _col_ * dispRange + ptrj * dispRange + d];
				//if (value > 60)  // 这里设的是截断SAD 
				//	value = 60;
				cost = value;
			}
		}
		double rate = double(i) / (row);  // showing the process
		cout << "cost aggregation finishing " << setprecision(2) << rate * 100 << "%" << endl;
	}
}

void costAggregationV4(float* dispvolume, float* CostVolume, int* ArmvolumeL, int* ArmvolumeR, int* ArmvolumeUp,
	int* ArmvolumeDown, int dispRange, int _row_, int _col_, int winSize)
{
	int col = _col_;
	int row = _row_;
	float* dispLayer = new float[col * row]();
	for (int d = 0; d < dispRange; ++d)
	{
		// 将某一层的视差值取出来放在dispLayer中
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				auto disp = dispvolume[i * col * dispRange + j * dispRange + d];
				dispLayer[i * col + j] = disp;
			}
		}
		//copyMakeBorder(imageR, imageR, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
		// 聚合这一层的dispLayer
		for (int i = 0; i < row; i++)  // 丢掉一行
		{
			for (int j = 0; j < col; j++)
			{
				// 注释的部分是zuo
				auto L = ArmvolumeL[i * col * dispRange + j * dispRange + d];
				auto R = ArmvolumeR[i * col * dispRange + j * dispRange + d];
				auto up = ArmvolumeUp[i * col * dispRange + j * dispRange + d];
				auto down = ArmvolumeDown[i * col * dispRange + j * dispRange + d];
				//auto L = ArmvolumeL[i * col  + j];
				//auto R = ArmvolumeR[i * col  + j];
				//auto up = ArmvolumeUp[i * col + j];
				//auto down = ArmvolumeDown[i * col  + j];
				float value = 0;
				int number = 0;
				for (int top = -up; top < down ; top++) // 从上到下
				{
					for (int left = -L; left < R; left++)
					{
						value = value +  dispLayer[(i + top) * col + j + left];
						number++;
					}
				}
				value = value / number;
				auto& cost = CostVolume[i * col * dispRange + j * dispRange + d];
				cost = value;
			}
		}
	}
}


void costAggregationV5(float* dispvolume, float* CostVolume, int* ArmvolumeL, int* ArmvolumeR, int* ArmvolumeUp,
	int* ArmvolumeDown, int dispRange, int _row_, int _col_, int winSize)
{
	int col = _col_;
	int row = _row_;
	float* dispLayer = new float[col * row]();
//#pragma omp parallel for
	for (int d = 0; d < dispRange; ++d)
	{
		// 将某一层的视差值取出来放在dispLayer中
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				auto disp = dispvolume[i * col * dispRange + j * dispRange + d];
				dispLayer[i * col + j] = disp;
			}
		}
		//copyMakeBorder(imageR, imageR, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
		// 聚合这一层的dispLayer
		for (int i = 0; i < row; i++)  // 丢掉一行
		{
			for (int j = 0; j < col; j++)
			{
				// 注释的部分是zuo
				auto L = ArmvolumeL[i * col  + j];
				auto R = ArmvolumeR[i * col  + j];
				auto up = ArmvolumeUp[i * col + j];
				auto down = ArmvolumeDown[i * col  + j];
				float value = 0;
				int number = 0;
				for (int top = -up; top <= down; top++) // 从上到下
				{
					for (int left = -L; left <=R; left++)
					{
						value = value + dispLayer[(i + top) * col + j + left];
						number++;
					}
				}
				value = value / number;
				auto& cost = CostVolume[i * col * dispRange + j * dispRange + d];
				cost = value;
			}
		}
	}
}


void costAggregationVertical(float* dispvolume, float* CostVolume, int* ArmvolumeL, int* ArmvolumeR, int* ArmvolumeUp,
	int* ArmvolumeDown, int dispRange, int _row_, int _col_, int winSize)
{
	int col = _col_;
	int row = _row_;
	float* dispLayer = new float[col * row]();
	//#pragma omp parallel for
	for (int d = 0; d < dispRange; ++d)
	{
		// 将某一层的视差值取出来放在dispLayer中
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				auto disp = dispvolume[i * col * dispRange + j * dispRange + d];
				dispLayer[i * col + j] = disp;
			}
		}
		//copyMakeBorder(imageR, imageR, winSize + 1, winSize + 1, winSize + 1, winSize + 1, BORDER_REPLICATE);
		// 聚合这一层的dispLayer
		for (int i = 0; i < row; i++)  // 丢掉一行
		{
			for (int j = 0; j < col; j++)
			{
				// 注释的部分是zuo
				auto L = ArmvolumeL[i * col + j];
				auto R = ArmvolumeR[i * col + j];
				auto up = ArmvolumeUp[i * col + j];
				auto down = ArmvolumeDown[i * col + j];
				float value = 0;
				int number = 0;
				for (int left= -L; left <= R; left++) // 从上到下
				{
					for (int u = -up; u <= down; u++)
					{
						value = value + dispLayer[(i + u) * col + j + left];
						number++;
					}
				}
				value = value / number;
				auto& cost = CostVolume[i * col * dispRange + j * dispRange + d];
				cost = value;
			}
		}
	}
}
