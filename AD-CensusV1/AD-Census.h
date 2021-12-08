#pragma once
#include <iostream>	
#include <opencv2/opencv.hpp>

#define SAFE_DELETE(p) {if(p)(delete[](p); p= nullptr;)}
using namespace std;
using namespace cv;

class AD_Census
{
public:
	AD_Census();
	void	Initialize(float* leftImage, float* rightImage, int dispRange, int row, int col, Mat &LImage, Mat& RImage, float sigmaC, float sigmaS);
	void ComputeADcensus();
	void ComputeADcensusRight();
	int HammingDistance(const int& x, const int& y	);
	void ComputeAD();
	void ComputeADRight();
	float* GetPtrLeft();
	float* GetPtrRight();
	void ComputeCensus9x7(float* leftImage,  float* rightImage);
	void ComputeCensus9x7Right(float* leftImage, float* rightImage);
	void WTA(float* leftdisp, float* rightDisp);
	~AD_Census();

private:
	Mat _leftImage, _rightImage;
	float* __leftImage__;
	float* __rightImage__;
	float* __leftDisp__;
	float* __rightDisp__;
	float* ADcostVolum;
	float* ADcostVolumRight;
	float* CensusVolum;
	float* CensusVolumRight;
	float* costVolume;
	float* costVolumeRight;
	int __dispRange__;
	int __row__;
	int __col__;
	float _sigmaC;
	float _sigmaS;
};

AD_Census::AD_Census()
{

}

float* AD_Census::GetPtrLeft()
{
	if (costVolume!=nullptr)
	{
		return &costVolume[0];
	}
	else
	{
		return nullptr;
	}
}

float* AD_Census::GetPtrRight()
{
	if (costVolumeRight != nullptr)
	{
		return &costVolumeRight[0];
	}
	else
	{
		return nullptr;
	}
}

// ���������AD��ԭ���еĻ�����һ�����ģ���Ϊ�����õ��ǻҶ�ֵȥ��
void AD_Census::ComputeAD()
{
	int row = __row__;
	int col = __col__;
	int dispRange = __dispRange__;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto leftValue = __leftImage__[i * col + j ];
			for (int d = 0; d < dispRange; d++)
			{
				auto& cost = ADcostVolum[i * col * dispRange + j * dispRange + d];
				if (j-d<0)
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

void AD_Census::ComputeADRight()
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
				auto& cost = ADcostVolumRight[i * col * dispRange + j * dispRange + d];
				if (j + d >=  col)
				{
					const float value = ADcostVolumRight[i * col * dispRange + j * dispRange + d - 1];
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

int AD_Census::HammingDistance(const int& x, const int& y)
{
	int value = 0, dist = x ^ y;
	while (dist)
	{
		++value;
		dist &= dist - 1;
	}
	return value;
}

void AD_Census::ComputeCensus9x7(float* leftImage, float* rightImage)
{
	//copyMakeBorder();  // �����ָ����ômakeboard������ʱ�����ˣ������������жϲ������ͺ���
	int row = __row__;
	int col = __col__;
	int dispRange = __dispRange__;
	for (int  i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto leftCenter = leftImage[i * col + j]; // ��ͼ��С������Ҷ�����
			int lastValue = 0;
			for (int d = 0; d < dispRange; d++)
			{
				uint64_t leftcensusValue = 0u;
				uint64_t rightcensusValue = 0u;
				float rigthCenter = 0; // ��ʼ�������ĵ�ֵ
				if(j-d<0)
					rigthCenter = rightImage[i * col + 0]; // �Ǿ͵�������ߵ�ֵ
				else
				{
					rigthCenter = rightImage[i * col + j - d];  // ��ͼС������ĻҶ����ģ���ȥƫ����
				}
				auto& cost = CensusVolum[i * col * dispRange + j * dispRange + d];
				// �ȼ���һС�������census 
				for (int r = -4; r <= 4; r++)
				{
					for (int c = -3; c <= 3; c++)
					{
						leftcensusValue <<= 1;
						rightcensusValue <<= 1;
						if (i + r < 0 || i + r >= row || j + c < 0 || j + c >= col)  // �������Խ���ֱֵ������Ϊ1�ͺ��ˣ��Ͳ�����������Ĳ�����
							continue;
						auto leftValue = leftImage[(i + r) * col + j + c];  // ��ͼ�����ĵ���Χ��ֵ
						float rightValue = 0;
						if (j + c - d < 0)
							rightValue = rightImage[(i + r) * col + 0];   // ��ͼ�����ĵ���Χ��ֵ
						else
						{
							rightValue = rightImage[(i + r) * col + j + c - d];  // ���d����Ҫ�ж�һ���ǲ��ǻᳬ����߽�
						}
						if (leftCenter>leftValue)
						{
							leftcensusValue += 1;
						}
						if (rigthCenter > rightValue)
						{
							rightcensusValue += 1;
						}
					}
				}  
				auto dist = leftcensusValue ^ rightcensusValue;  // ���㺺������ 
				while (dist)
				{
					lastValue++;
					dist &= dist - 1;
				}
				cost = lastValue;
				lastValue = 0;
			}
		}
	}
}


void AD_Census::ComputeCensus9x7Right(float* leftImage, float* rightImage)
{
	//copyMakeBorder();  // �����ָ����ômakeboard������ʱ�����ˣ������������жϲ������ͺ���
	int row = __row__;
	int col = __col__;
	int dispRange = __dispRange__;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto rightCenter = rightImage[i * col + j]; // ��ͼ��С������Ҷ�����
			int lastValue = 0;
			for (int d = 0; d < dispRange; d++)
			{
				uint64_t leftcensusValue = 0u;
				uint64_t rightcensusValue = 0u;
				float leftCenter = 0; // ��ʼ�������ĵ�ֵ
				if (j + d >= col)
					leftCenter = leftImage[i * col + col -1]; // �Ǿ͵�������ߵ�ֵ
				else
				{
					leftCenter = leftImage[i * col + j + d];  // ��ͼС������ĻҶ����ģ���ȥƫ����
				}
				auto& cost = CensusVolumRight[i * col * dispRange + j * dispRange + d];
				// �ȼ���һС�������census 
				for (int r = -4; r <= 4; r++)
				{
					for (int c = -3; c <= 3; c++)
					{
						leftcensusValue <<= 1;
						rightcensusValue <<= 1;
						if (i + r < 0 || i + r >= row || j + c < 0 || j + c >= col)  // �������Խ���ֱֵ������Ϊ0�ͺ��ˣ��Ͳ�����������Ĳ�����
							continue;
						auto rightValue = rightImage[(i + r) * col + j + c];  // ��ͼ�����ĵ���Χ��ֵ
						float leftValue = 0;
						if (j + c + d >= col)
							leftValue = leftImage[(i + r) * col + 0];   // ��ͼ�����ĵ���Χ��ֵ
						else
						{
							leftValue = leftImage[(i + r) * col + j + c + d];  // ���d����Ҫ�ж�һ���ǲ��ǻᳬ����߽�
						}
						if (leftCenter > leftValue)
						{
							leftcensusValue += 1;
						}
						if (rightCenter > rightValue)
						{
							rightcensusValue += 1;
						}
					}
				}
				auto dist = leftcensusValue ^ rightcensusValue;  // ���㺺������ 
				while (dist)
				{
					lastValue++;
					dist &= dist - 1;
				}
				cost = lastValue;
				lastValue = 0;
			}
		}
	}
}

void AD_Census::ComputeADcensus()
{
	int row = __row__;
	int col = __col__;
	int dispRange = __dispRange__;
	ComputeAD();
	ComputeCensus9x7(__leftImage__, __rightImage__);
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; j++)
		{
			for (int d = 0; d < dispRange; d++)
			{
				auto& cost = costVolume[i * col * dispRange + j * dispRange + d];
				const float AD = ADcostVolum[i * col * dispRange + j * dispRange + d];
				const float census = CensusVolum[i * col * dispRange + j * dispRange + d];
				auto ADvalue = 1 - exp(-(AD / _sigmaC));
				auto censusValue = 1 - exp(-(census / _sigmaS));
				cost = ADvalue + censusValue; // һ������
				//cost = censusValue;
			}
		}
	}
}

void AD_Census::ComputeADcensusRight()
{
	int row = __row__;
	int col = __col__;
	int dispRange = __dispRange__;
	ComputeADRight();
	ComputeCensus9x7Right(__leftImage__, __rightImage__);
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; j++)
		{
			for (int d = 0; d < dispRange; d++)
			{
				auto& cost = costVolumeRight[i * col * dispRange + j * dispRange + d];
				const float AD = ADcostVolumRight[i * col * dispRange + j * dispRange + d];
				const float census = CensusVolumRight[i * col * dispRange + j * dispRange + d];
				auto ADvalue = 1 - exp(-(AD / _sigmaC));
				auto censusValue = 1 - exp(-(census / _sigmaS));
				cost = ADvalue + censusValue; // һ������
			}
		}
	}
}



void AD_Census::Initialize(float* leftImage, float* rightImage, int dispRange, int row, int col, Mat& LImage, Mat& RImage, float sigmaC, float sigmaS)
{
	__leftImage__ = leftImage;
	__rightImage__ = rightImage;
	__dispRange__ = dispRange;
	__row__ = row;
	__col__ = col;
	_leftImage = LImage;
	_rightImage = RImage;
	_sigmaC = sigmaC;
	_sigmaS = sigmaS;

	int size = row * col;
	__leftDisp__ = new float[size]();
	__rightDisp__ = new float[size]();
	ADcostVolum = new float[size*dispRange]();
	ADcostVolumRight = new float[size * dispRange]();
	CensusVolum = new float[size*dispRange]();
	CensusVolumRight = new float[size * dispRange]();
	costVolume = new float[size*dispRange]();
	costVolumeRight = new float[size * dispRange]();

}

void AD_Census::WTA(float* leftdisp, float* rightDisp )
{
	int col = __col__;
	int row = __row__;
	int dispRange = __dispRange__;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			auto cost = costVolume[i * col * dispRange + j * dispRange + 0];
			auto costRight = costVolumeRight[i * col * dispRange + j * dispRange + 0];
			float disparity = 0;
			float disparityRight = 0;
			for (int d = 1; d < dispRange; d++)
			{
				auto value = costVolume[i * col * dispRange + j * dispRange + d];
				auto valueRight = costVolumeRight[i * col * dispRange + j * dispRange + d];
				if (cost > value)
				{
					disparity = d;
					cost = value;
				}
				if (costRight > valueRight)
				{
					costRight = valueRight;
					disparityRight = d;
				}
			}
			auto& costDispRight = rightDisp[i * col + j];
			auto& costDisp =	leftdisp[i * col + j];
			costDisp = disparity;
			costDispRight = disparityRight;
		}
	}
}

AD_Census::~AD_Census()
{

}